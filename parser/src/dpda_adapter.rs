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
        Ok(())
    }
}

/// Compile the CGrammar to a PDA machine (the RTN construction).
pub fn compile_pda(grm: &CGrammar) -> Result<pushdown_rs::machine::PdaMachine, CfgError> {
    let adapter = PdaGrammar::new(grm);
    pushdown_rs::compile(&adapter)
}

/// Compile the CGrammar's PDA as a CUDA package (the bitvec + the source
/// primitives). This is the H2D payload for the attention-rs kernels.
pub fn export_pda_package(grm: &CGrammar) -> Result<pushdown_rs::cuda::CudaPackage, CfgError> {
    let machine = compile_pda(grm)?;
    pushdown_rs::cuda::CudaPackage::from_machine(&machine).map_err(|e| {
        CfgError::Other(format!("the CUDA package export failed: {e}"))
    })
}