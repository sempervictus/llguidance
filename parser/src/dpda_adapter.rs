//! The adapter: implement the pushdown_rs::Grammar trait for the llguidance
//! CGrammar (the compiled grammar). This is the bridge that lets the
//! pushdown-rs core compile the llguidance's grammars to PDAs.
//!
//! The critical detail (the lesson from the global/local index bug): the
//! terminal_id/nonterminal_id must return the LOCAL indices (the 0..num_terminals,
//! the 0..num_nonterminals), NOT the global symbol IDs (the 0..num_symbols).
//! The RTN compilation numbers the states by the local index.
//!
//! Performance: the CGrammar uses a flat CSymIdx over ALL symbols (terminals +
//! nonterminals mixed). The wrapper precomputes the global->local maps once,
//! giving O(1) ID lookups during compilation (vs the O(n) scan per call).

use pushdown_rs::compile::{CfgError, Grammar};

use crate::earley::{CGrammar, CSymIdx};
use derivre::RegexAst;

/// A wrapper around &CGrammar that precomputes the global->local symbol index
/// maps for O(1) terminal_id/nonterminal_id lookups during RTN compilation.
pub struct PdaGrammar<'a> {
    grm: &'a CGrammar,
    /// global CSymIdx (0..num_symbols) -> local terminal index (0..num_terminals), or None
    global_to_local_terminal: Vec<Option<u32>>,
    /// global CSymIdx (0..num_symbols) -> local nonterminal index (0..num_nonterminals), or None
    global_to_local_nonterminal: Vec<Option<u32>>,
    #[allow(dead_code)]
    num_terminals: u32,
    #[allow(dead_code)]
    num_nonterminals: u32,
    start_local: u32,
}

impl<'a> PdaGrammar<'a> {
    pub fn new(grm: &'a CGrammar) -> Self {
        let num_syms = grm.num_symbols();
        let mut global_to_local_terminal = vec![None; num_syms];
        let mut global_to_local_nonterminal = vec![None; num_syms];
        let mut term_count = 0u32;
        let mut nt_count = 0u32;
        for i in 0..num_syms {
            let s = CSymIdx::new_checked(i);
            if grm.is_terminal(s) {
                global_to_local_terminal[i] = Some(term_count);
                term_count += 1;
            } else {
                global_to_local_nonterminal[i] = Some(nt_count);
                nt_count += 1;
            }
        }
        let start_local = global_to_local_nonterminal
            .get(grm.start().as_index())
            .copied()
            .flatten()
            .unwrap_or(0);
        PdaGrammar {
            grm,
            global_to_local_terminal,
            global_to_local_nonterminal,
            num_terminals: term_count,
            num_nonterminals: nt_count,
            start_local,
        }
    }

    pub fn grammar(&self) -> &CGrammar {
        self.grm
    }
}

impl<'a> Grammar for PdaGrammar<'a> {
    type Nonterminal = CSymIdx;
    type Terminal = CSymIdx;
    type Symbol = CSymIdx;

    fn nonterminals(&self) -> Vec<CSymIdx> {
        (0..self.grm.num_symbols())
            .filter(|&i| self.global_to_local_nonterminal[i].is_some())
            .map(CSymIdx::new_checked)
            .collect()
    }

    fn terminals(&self) -> Vec<CSymIdx> {
        (0..self.grm.num_symbols())
            .filter(|&i| self.global_to_local_terminal[i].is_some())
            .map(CSymIdx::new_checked)
            .collect()
    }

    fn start(&self) -> CSymIdx {
        self.grm.start()
    }

    fn productions(&self) -> Vec<(CSymIdx, Vec<CSymIdx>)> {
        let mut prods = Vec::new();
        for i in 0..self.grm.num_symbols() {
            let s = CSymIdx::new_checked(i);
            if self.grm.is_terminal(s) {
                continue;
            }
            for &rule in self.grm.rules_of(s) {
                let rhs = self.grm.rhs_symbols(rule).to_vec();
                prods.push((s, rhs));
            }
        }
        prods
    }

    fn is_terminal(&self, sym: &CSymIdx) -> bool {
        self.grm.is_terminal(*sym)
    }

    fn terminal_id(&self, sym: &CSymIdx) -> Option<u32> {
        self.global_to_local_terminal.get(sym.as_index()).copied().flatten()
    }

    fn terminal_name(&self, sym: &CSymIdx) -> String {
        // The CGrammar terminal's name (the "reasoning_block", the "text", the
        // "tool_call", ...). The consumer (the llguidance mask builder) maps this
        // name back to the terminal's token range (the LexemeSpec.token_ranges), so
        // the PDA input bit is identified with the actual tokenizer lexeme.
        self.grm.sym_name(*sym).to_string()
    }

    fn nonterminal_id(&self, sym: &CSymIdx) -> Option<u32> {
        self.global_to_local_nonterminal.get(sym.as_index()).copied().flatten()
    }

    fn nonterminal_index(&self, nt: &CSymIdx) -> u32 {
        self.global_to_local_nonterminal.get(nt.as_index()).copied().flatten().unwrap_or(0)
    }

    fn start_id(&self) -> u32 {
        self.start_local
    }

    fn validate(&self) -> Result<(), CfgError> {
        // The RTN compilation is CFG-only. Parametric grammars (the 64-bit rule
        // conditions evaluated at Earley-item time) cannot be compiled to a PDA.
        // Reject them here so the PDA path is never taken for aetric grammars.
        if self.grm.parametric() {
            return Err(CfgError::Other(
                "parametric grammars cannot be compiled to a PDA (the RTN is CFG-only)".into(),
            ));
        }
        Ok(())
    }
}

/// Compile the CGrammar to a PDA machine (the RTN construction).
pub fn compile_pda(grm: &CGrammar) -> Result<pushdown_rs::machine::PdaMachine, CfgError> {
    let adapter = PdaGrammar::new(grm);
    pushdown_rs::compile(&adapter)
}

/// The terminal-to-token bridge: for each PDA local terminal ID `a`
/// (0..num_inputs), the set of vocabulary token IDs that terminal covers.
///
/// Derived from `CGrammar.terminal_token_ranges` (the precomputed
/// `LexemeSpec.token_ranges`). When a terminal's ranges are empty, the
/// token set is computed from the tokenizer's trie (the `TokenSpanner`
/// approach): iterate over the vocabulary and check which tokens' byte
/// sequences match the terminal's regex.
pub fn terminal_token_map(grm: &CGrammar, tok_env: &crate::toktrie::TokEnv) -> Vec<Vec<u32>> {
    terminal_token_map_dfa(grm, tok_env, None)
}

/// The full bridge computation with an optional DFA (the `RegexVec`) for
/// matching the complex `RegexAst` terminals (the `Or`, the `Regex`, the
/// `ExprRef`). When the DFA is provided, the bridge entries for the complex
/// terminals are computed by driving the DFA over each vocabulary token's
/// byte sequence and checking if the terminal's lexeme is in the accepting set.
pub fn terminal_token_map_dfa(
    grm: &CGrammar,
    tok_env: &crate::toktrie::TokEnv,
    dfa: Option<&mut crate::earley::regexvec::RegexVec>,
) -> Vec<Vec<u32>> {
    let adapter = PdaGrammar::new(grm);
    let terminals = adapter.terminals(); // the CSymIdx list, in local-ID order
    let trie = tok_env.tok_trie();
    let vocab_size = trie.vocab_size();
    let num_terminals = terminals.len();

    // Two-pass partition:
    // Pass 1: assign specific tokens to their terminals (the special tokens via
    //   token_ranges, the single-byte terminals via token_id, the multi-byte
    //   terminals via greedy_tokenize). Mark catch-all terminals (the negated
    //   ranges) for pass 2.
    // Pass 2: assign the remaining tokens (the complement of pass 1) to the
    //   catch-all terminals. This ensures disjointness + totality.

    let mut bridge: Vec<Vec<u32>> = vec![Vec::new(); num_terminals];
    let mut is_catch_all = vec![false; num_terminals];
    let mut claimed: Vec<bool> = vec![false; vocab_size];

    for (i, &csym) in terminals.iter().enumerate() {
        let ranges = grm.terminal_token_ranges(csym);
        if !ranges.is_empty() {
            // Specific: the token_ranges are populated (the special tokens).
            for range in ranges.iter() {
                for tok in range.clone() {
                    let idx = tok as usize;
                    if idx < vocab_size && !claimed[idx] {
                        bridge[i].push(tok);
                        claimed[idx] = true;
                    }
                }
            }
            continue;
        }

        // No token_ranges: check the terminal's display name for the range pattern.
        let sym_name = grm.sym_name(csym);
        let name_bytes = sym_name.as_bytes();

        if name_bytes.starts_with(b"<[^") && name_bytes.ends_with(b">") {
            // Catch-all: the negated range terminal. Its token set is the
            // complement of all specific tokens (computed in pass 2).
            is_catch_all[i] = true;
            continue;
        }

        if name_bytes.starts_with(b"<[") && name_bytes.ends_with(b">") {
            // Positive range terminal: the token set is the specified ranges.
            // Parse the ranges from the display name (e.g., "<[0-248044,248046-248057]>").
            let inner = std::str::from_utf8(&name_bytes[2..name_bytes.len() - 1]).unwrap_or("");
            for range_str in inner.split(',') {
                let range_str = range_str.trim();
                if let Some(dash_pos) = range_str.find('-') {
                    let start: u32 = range_str[..dash_pos].parse().unwrap_or(0);
                    let end: u32 = range_str[dash_pos + 1..].parse().unwrap_or(0);
                    for tok in start..=end {
                        let idx = tok as usize;
                        if idx < vocab_size && !claimed[idx] {
                            bridge[i].push(tok);
                            claimed[idx] = true;
                        }
                    }
                } else if !range_str.is_empty() {
                    if let Ok(tok) = range_str.parse::<u32>() {
                        let idx = tok as usize;
                        if idx < vocab_size && !claimed[idx] {
                            bridge[i].push(tok);
                            claimed[idx] = true;
                        }
                    }
                }
            }
            continue;
        }

        // Single-byte or multi-byte terminal: use the RegexAst to extract the byte pattern.
        let lexeme_idx = grm.sym_data(csym).lexeme;
        if let Some(lex) = lexeme_idx {
            let lexeme_spec = grm.lexer_spec().lexeme_spec(lex);
            // Extract the byte pattern from the RegexAst (the literal string, the
            // single byte, or the concatenation of literals).
            let byte_pattern = extract_byte_pattern(&lexeme_spec.rx);
            if let Some(bytes) = byte_pattern {
                if bytes.len() == 1 {
                    // Single-byte terminal: the exact byte match.
                    if let Some(tok_id) = trie.token_id(&bytes) {
                        let idx = tok_id as usize;
                        if idx < vocab_size && !claimed[idx] {
                            bridge[i].push(tok_id);
                            claimed[idx] = true;
                        }
                    }
                } else if bytes.len() > 1 {
                    // Multi-byte terminal: the greedy tokenization.
                    let token_ids = trie.greedy_tokenize(&bytes);
                    for tok_id in token_ids {
                        let idx = tok_id as usize;
                        if idx < vocab_size && !claimed[idx] {
                            bridge[i].push(tok_id);
                            claimed[idx] = true;
                        }
                    }
                }
            }
        }
    }

    // Pass 1.5: DFA matching for the complex RegexAst terminals (the Or, the
    // Regex, the ExprRef). When the DFA is provided, drive it over each
    // vocabulary token's byte sequence and check if the terminal's lexeme is
    // in the accepting set. This fills the bridge entries that Pass 1 left
    // empty (the terminals with complex regexes).
    if let Some(dfa) = dfa {
        for (i, &csym) in terminals.iter().enumerate() {
            if !bridge[i].is_empty() {
                continue; // already filled by Pass 1
            }
            let lexeme_idx = grm.sym_data(csym).lexeme;
            let Some(lex) = lexeme_idx else { continue }; // the NULL terminal
            // Build the selected lexeme set (just our terminal's lexeme).
            let mut selected = grm.lexer_spec().alloc_lexeme_set();
            selected.add(lex);
            // Drive the DFA over each unclaimed vocabulary token.
            let initial = dfa.initial_state(&selected);
            for tok_id in 0..vocab_size as u32 {
                if claimed[tok_id as usize] {
                    continue; // already assigned to another terminal
                }
                let token_str = trie.token_str(tok_id);
                let token_bytes = token_str.as_bytes();
                let mut state = initial;
                for &byte in token_bytes {
                    state = dfa.transition(state, byte);
                }
                // Check if our terminal's lexeme is in the accepting set.
                let desc = dfa.state_desc(state);
                if desc.greedy_accepting.contains(lex) || desc.lazy_accepting.contains(lex) {
                    bridge[i].push(tok_id);
                    claimed[tok_id as usize] = true;
                }
            }
        }
    }

    // Pass 2: assign the remaining tokens to the catch-all terminals.
    // If there are multiple catch-all terminals, the first one gets the
    // remainder (the others get empty sets, which is correct: only one is
    // the catch-all, the rest are subsumed by it).
    let mut catch_all_assigned = false;
    for i in 0..num_terminals {
        if !is_catch_all[i] || catch_all_assigned {
            continue;
        }
        for tok in 0..vocab_size as u32 {
            if !claimed[tok as usize] {
                bridge[i].push(tok);
                claimed[tok as usize] = true;
            }
        }
        catch_all_assigned = true;
    }

    // Sort + dedup each bridge entry.
    for tokens in bridge.iter_mut() {
        tokens.sort();
        tokens.dedup();
    }

    bridge
}

/// The algorithmic bridge contract: the terminal-to-token mapping is
/// sound for the PDA mask iff ALL three properties hold:
///
/// 1. NO FALLBACK: every terminal has non-empty `token_ranges` (the
///    identity fallback is not used).
/// 2. DISJOINT: no vocabulary token ID appears in two different
///    terminals' ranges.
/// 3. TOTAL: every vocabulary token ID in [0, vocab_size) is covered
///    by at least one terminal's range.
///
/// When all three hold, the PDA mask (the `mask_at_cfg` lifted through
/// the bridge) is a precise token-level mask equivalent to the legacy
/// Earley `compute_bias`. When any fails, the PDA mask is an
/// over- or under-approximation and must not replace the legacy path.
pub fn bridge_is_exact(grm: &CGrammar, vocab_size: usize, tok_env: &crate::toktrie::TokEnv) -> bool {
    bridge_is_exact_dfa(grm, vocab_size, tok_env, None)
}

/// The full bridge exactness check with an optional DFA (the `RegexVec`) for
/// matching the complex `RegexAst` terminals. When the DFA is provided, the
/// bridge entries for the complex terminals are computed by driving the DFA
/// over each vocabulary token's byte sequence.
pub fn bridge_is_exact_dfa(
    grm: &CGrammar,
    vocab_size: usize,
    tok_env: &crate::toktrie::TokEnv,
    dfa: Option<&mut crate::earley::regexvec::RegexVec>,
) -> bool {
    let bridge = terminal_token_map_dfa(grm, tok_env, dfa);
    let trie = tok_env.tok_trie();
    // The EOS tokens are not part of the grammar (they're the end-of-sequence
    // markers). Exclude them from the totality check.
    let eos_tokens: std::collections::HashSet<u32> = trie.eos_tokens().iter().copied().collect();

    // Property 2: disjointness over the full vocabulary.
    let mut covered = vec![false; vocab_size];
    for token_list in bridge.iter() {
        for &tok in token_list {
            let idx = tok as usize;
            if idx >= vocab_size {
                continue;
            }
            if covered[idx] {
                return false; // Property 2 violated: token in two terminals
            }
            covered[idx] = true;
        }
    }
    // Property 3: every NON-EOS token is covered.
    covered.iter().enumerate().all(|(idx, &c)| c || eos_tokens.contains(&(idx as u32)))
}

/// Compile the CGrammar's PDA as a CUDA package (the bitvec + the source
/// primitives). This is the H2D payload for the attention-rs kernels.
pub fn export_pda_package(grm: &CGrammar) -> Result<pushdown_rs::cuda::CudaPackage, CfgError> {
    let machine = compile_pda(grm)?;
    pushdown_rs::cuda::CudaPackage::from_machine(&machine).map_err(|e| {
        CfgError::Other(format!("the CUDA package export failed: {e}"))
    })
}

/// Extract a fixed byte pattern from a `RegexAst` (the simple cases: the
/// `Literal`, the `Byte`, the `Concat` of literals/bytes). Returns `None`
/// for complex regexes (the `Or`, the `Regex`, the `AndRef`) that require
/// the DFA matching approach (the next increment).
fn extract_byte_pattern(ast: &RegexAst) -> Option<Vec<u8>> {
    match ast {
        RegexAst::Literal(s) => Some(s.as_bytes().to_vec()),
        RegexAst::Byte(b) => Some(vec![*b]),
        RegexAst::Concat(parts) => {
            let mut result = Vec::new();
            for part in parts {
                let bytes = extract_byte_pattern(part)?;
                result.extend(bytes);
            }
            Some(result)
        }
        _ => None, // the complex regexes (the Or, the Regex, the ExprRef) need the DFA
    }
}