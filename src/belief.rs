use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::utils::logsumexp;
use crate::hmm::{HMM, get_test_hmm};
use crate::potts::get_test_potts;
use crate::tree::get_test_tree;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType { Latent, Emission }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType { Emission, Transition, Prior, Custom }

#[derive(Debug, Clone)]
pub struct Factor {
    pub id: usize,
    pub factor_type: FactorType,
    pub variables: Vec<usize>,
    pub table: Rc<Vec<f64>>,
}

// NB Forney factor graph with variables and factors connected by edges.
pub struct FactorGraph {
    pub factors: Vec<Factor>,
    pub var_adj: HashMap<usize, Vec<usize>>, // var_id -> list of adjacent factor_ids.  Deprecate?
    pub domain_size: usize,
    pub num_vars: usize,
    pub var_to_factor: HashMap<(usize, usize), Vec<f64>>, // (var_id, factor_id) -> message for s in domain.
    pub factor_to_var: HashMap<(usize, usize), Vec<f64>>, // (factor_id, var_id) -> message for s in domain.
    pub rng: StdRng,
}

#[derive(Clone, Copy)]
enum WorkItem {
    VarToFactor { var_id: usize, factor_id: usize },
    FactorToVar { factor_id: usize, var_id: usize },
}

impl FactorGraph {
    pub fn new(num_vars: usize, domain_size: usize) -> Self {
        Self {
            factors: Vec::new(),
            var_adj: HashMap::new(),
            domain_size,
            num_vars,
            var_to_factor: HashMap::new(),
            factor_to_var: HashMap::new(),
            rng: StdRng::seed_from_u64(42),
        }
    }
    
    pub fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    pub fn add_factor(&mut self, variables: Vec<usize>, table: Vec<f64>, factor_type: FactorType) {
        // NB constructor creates a new (reference counted) table so that it is not shared.
        self.add_shared_factor(variables, Rc::new(table), factor_type);
    }

    pub fn add_shared_factor(&mut self, variables: Vec<usize>, table: Rc<Vec<f64>>, factor_type: FactorType) {
        let id = self.factors.len();

        // NB for each variable in this factor, add the factor to its adjacency.
        for &v in &variables {
            self.var_adj.entry(v).or_default().push(id);
        }

        // NB reference count the factor table for memory efficiency when shared. 
        self.factors.push(Factor { id, variables, table, factor_type });
    }

    // NB calculate the marginals for each variable as incoming Factor->Var messages produce & and normalize. 
    pub fn calculate_marginals(&self) -> Vec<Vec<f64>> {
        let mut marginals = Vec::with_capacity(self.num_vars);
        for v in 0..self.num_vars {
            let mut log_prob = vec![0.0; self.domain_size];
            // NB marginal belief at a variable node v is the sum of all ln prob. messages from adjacent factors to v.  
            //    See eqn. (14.16) in Mezard.
            if let Some(n_fids) = self.var_adj.get(&v) {
                for &fid in n_fids {
                    if let Some(msg) = self.factor_to_var.get(&(fid, v)) {
                        for i in 0..self.domain_size {
                            log_prob[i] += msg[i];
                        }
                    }
                }
            }
            normalize_log_msg(&mut log_prob);
            marginals.push(log_prob);
        }
        marginals
    }

    pub fn run_belief_propagation(&mut self, max_iters: usize, tolerance: f64, dropout_rate: f64) -> Vec<Vec<f64>>{
        let mut queue = VecDeque::new();

        for factor in &self.factors {
            for &var in &factor.variables {
                self.var_to_factor.entry((var, factor.id)).or_insert_with(|| vec![0.0; self.domain_size]);
                self.factor_to_var.entry((factor.id, var)).or_insert_with(|| vec![0.0; self.domain_size]);
                
                queue.push_back(WorkItem::VarToFactor { var_id: var, factor_id: factor.id });
                queue.push_back(WorkItem::FactorToVar { factor_id: factor.id, var_id: var });
            }
        }

        // NB set a reasonable upper bound on iterations to prevent infinite loops in pathological cases.
        let limit = 200_000.min(max_iters * self.factors.len().max(1) * 10).max(1000);
        let mut iters = 0;

        // NB pop one Var->Factor or Factor->Var message update to process.
        while let Some(item) = queue.pop_front() {
            iters += 1;
            if iters > limit { break; }

            match item {
                WorkItem::VarToFactor { var_id, factor_id } => {
                    // NB message of var j to factor a is the product of all incoming messages to var j except from a.
                    //    In log space, this is a sum.  See eqn. (14.14) in Mezard.
                    let mut incoming = vec![0.0; self.domain_size];

                    // NB 
                    if let Some(neighbors) = self.var_adj.get(&var_id) {
                         for i in 0..self.domain_size {
                             let mut sum = 0.0;

                             // NB all neighbouring factors except the target factor, a.
                             for &n_fid in neighbors {
                                 if n_fid != factor_id {
                                     if let Some(msg) = self.factor_to_var.get(&(n_fid, var_id)) {
                                         sum += msg[i];
                                     }
                                 }
                             }
                             incoming[i] = sum;
                        }
                    }
                    normalize_log_msg(&mut incoming);

                    // NB get the old message and check for update magnitude.
                    let entry = self.var_to_factor.get_mut(&(var_id, factor_id)).unwrap();

                    // NB diff is the L1 distance between the new message and the old message. If it's above tolerance,
                    //    update & schedule neighbors.
                    let diff: f64 = incoming.iter().zip(entry.iter()).map(|(a, b)| (a - b).abs()).sum();
                    
                    if diff > tolerance {
                        *entry = incoming;

                        let factor = &self.factors[factor_id];
                        for &n_var in &factor.variables {
                            if n_var != var_id {
                                // NB support for target factor a over domain has updated, schedule all neighboring variables.
                                queue.push_back(WorkItem::FactorToVar { factor_id, var_id: n_var });
                            }
                        }
                    }
                }
                WorkItem::FactorToVar { factor_id, var_id } => {
                    // NB Sum over possible adjacent variable configurations for this factor (with target j set) of product of incoming messages
                    //    from variables adjacent to a that are not target variable j and this factor @ the current config.  See eqn. (14.15) in Mezard.
                    let factor = &self.factors[factor_id];
                    let target_idx = factor.variables.iter().position(|&x| x == var_id).unwrap();
                    
                    // NB collect incoming messages from variables adjacent to this factor except the target variable.
                    //    defaults to the uniform log message if no incoming message exists (e.g. at initialization).
                    let incoming_msgs: Vec<Vec<f64>> = factor.variables.iter().map(|&v| {
                        self.var_to_factor.get(&(v, factor_id)).cloned().unwrap_or_else(|| vec![0.0; self.domain_size])
                    }).collect();

                    let mut new_msg = vec![0.0; self.domain_size];

                    // NB Apply dropout if requested: pass uniform message with probability p.
                    if self.rng.random::<f64>() >= dropout_rate {
                        new_msg = compute_factor_message(factor, target_idx, self.domain_size, &incoming_msgs);
                    }

                    normalize_log_msg(&mut new_msg);

                    // NB get the old message and check for update magnitude.
                    let entry = self.factor_to_var.get_mut(&(factor_id, var_id)).unwrap();
                    let diff: f64 = new_msg.iter().zip(entry.iter()).map(|(a, b)| (a - b).abs()).sum();

                    if diff > tolerance {
                        *entry = new_msg;

                        // NB schedule adjacent variables to this factor.
                        if let Some(neighbors) = self.var_adj.get(&var_id) {
                            for &n_fid in neighbors {
                                if n_fid != factor_id {
                                    queue.push_back(WorkItem::VarToFactor { var_id, factor_id: n_fid });
                                }
                            }
                        }
                    }
                }
            }
        }

        self.calculate_marginals()
    }
}

fn normalize_log_msg(msg: &mut [f64]) {
    let lse = logsumexp(msg);
    for x in msg { *x -= lse; }
}

fn next_assignment(assignment: &mut [usize], domain_size: usize, skip: usize) -> bool {
    // NB increment the assignment vector with each spin taking 0..domain_size,
    //    skipping a target variable.
    for (j, val) in assignment.iter_mut().enumerate() {
        if j == skip { continue; }
        *val += 1;
        if *val < domain_size { return true; }
        *val = 0;
    }
    false
}

fn get_factor_table_index(assignment: &[usize], domain_size: usize) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    // NB row-major: last index corresponds to stride 1.
    for &a in assignment.iter().rev() {
        idx += a * stride;
        stride *= domain_size;
    }
    idx
}

fn compute_factor_message(
    factor: &Factor,
    target_var_idx: usize, // Index in factor.variables
    domain_size: usize,
    incoming_messages: &[Vec<f64>]
) -> Vec<f64> {
    let mut messages = vec![0.0; domain_size];
    
    // NB we need the message component for the target variable fixed to each in its domain.
    for target_state in 0..domain_size {
        // NB we sum over all configurations of the other variables in this factor, multiplying
        //    the factor table value for that configuration with the incoming messages for those
        //    variables at that configuration.  In log space, this is a sum of the factor table
        //    value and the incoming messages, followed by a logsumexp over all configurations.
        let mut current_assignment = vec![0; factor.variables.len()];

        // NB analagous to the parent logic in pruning.
        current_assignment[target_var_idx] = target_state;

        // NB construct the message for target_var in target_state
        let mut log_terms = Vec::new();

        // NB sum over all other variables' configurations
        loop {
            // NB compute table index (row-major) for this factor at the current assignment.
            let idx = get_factor_table_index(&current_assignment, domain_size);

            let mut term = factor.table[idx];
            for (j, msg) in incoming_messages.iter().enumerate() {
                if j != target_var_idx {
                    term += msg[current_assignment[j]]; 
                }
            }
            log_terms.push(term);

            // NB exhausted all configurations of other variables.
            if !next_assignment(&mut current_assignment, domain_size, target_var_idx) {
                break;
            }
        }

        // NB to be normalized later.
        messages[target_state] = logsumexp(&log_terms);
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmm::HMM;
    use crate::potts::get_test_potts;
    use crate::tree::get_test_tree;

    #[test]
    fn test_rng() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let nums: Vec<i32> = (1..100).collect();
    
        let secret = nums.choose(&mut rng);

        assert!(secret == Some(&14));
    }

    #[test]
    fn test_normalize_log_msg() {
        let mut msg = vec![1.0, 2.0, 3.0];
        normalize_log_msg(&mut msg);
        let lse = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();

        assert!((msg[0] - (1.0 - lse)).abs() < 1e-6);
        assert!((msg[1] - (2.0 - lse)).abs() < 1e-6);
        assert!((msg[2] - (3.0 - lse)).abs() < 1e-6);

    }

    #[test]
    fn test_factor_table_indexing() {
        let domain_size = 3;
        
        // assignment [0, 0] -> 0
        assert_eq!(get_factor_table_index(&[0, 0], domain_size), 0);
        // assignment [0, 1] -> 0*3 + 1 = 1
        assert_eq!(get_factor_table_index(&[0, 1], domain_size), 1);
        // assignment [0, 2] -> 2
        assert_eq!(get_factor_table_index(&[0, 2], domain_size), 2);
        // assignment [1, 0] -> 1*3 + 0 = 3
        assert_eq!(get_factor_table_index(&[1, 0], domain_size), 3);
        // assignment [2, 2] -> 2*3 + 2 = 8
        assert_eq!(get_factor_table_index(&[2, 2], domain_size), 8);

        // 3 vars [0, 1, 2], domain 3
        // 0 * 9 + 1 * 3 + 2 * 1 = 3 + 2 = 5
        assert_eq!(get_factor_table_index(&[0, 1, 2], domain_size), 5);
    }

    #[test]
    fn test_next_assignment() {
        let mut assignment = vec![0, 0, 0];
        let domain_size = 2;
        let skip = 1;

        let mut results = Vec::new();
        loop {
            results.push(assignment.clone());
            if !next_assignment(&mut assignment, domain_size, skip) {
                break;
            }
        }

        assert_eq!(results, vec![
            vec![0, 0, 0],
            vec![1, 0, 0],
            vec![0, 0, 1],
            vec![1, 0, 1],
        ]);
    }

    #[test]
    fn test_chain_marginals() {
        // NB exact inference is 2^N=32 configs for N=5.
        let n_states: usize = 3;
        let chain_len: usize = 10;

        let (hmm, obs) = get_test_hmm(n_states, chain_len);
        let (prior, trans, emit) = (hmm.prior.clone(), hmm.trans.clone(), hmm.emit.clone());

        let exact_marginals = hmm.marginals(&obs);
        
        // NB Construct the factor graph.
        let mut fg = FactorGraph::new(chain_len, n_states);

        // NB prior factor on the first variable in the chain.
        fg.add_factor(vec![0], prior.iter().map(|x| x.ln()).collect(), FactorType::Prior);

        // NB Transition factors (shared memory)
        let trans_log: Vec<f64> = trans.iter().map(|x| x.ln()).collect();
        let trans_log_rc = Rc::new(trans_log);
        for t in 0..chain_len-1 {
            fg.add_shared_factor(vec![t, t+1], trans_log_rc.clone(), FactorType::Transition);
        }

        // Emissions (Unary factors)
        for t in 0..chain_len {
            let mut emit_log = vec![0.0; n_states];
            for k in 0..n_states {
                emit_log[k] = emit[k * 2 + obs[t]].ln(); 
            }
            fg.add_factor(vec![t], emit_log, FactorType::Emission);
        }

        println!("Running Belief Propagation on {}-chain", chain_len);
        let bp_marginals_log =fg.run_belief_propagation(50, 1e-6, 0.0);
        
        assert_eq!(bp_marginals_log.len(), chain_len);
        
        for t in 0..chain_len {
             // NB marginals from Forward-Backward via HMM struct.
             let mj = &exact_marginals[t];

             // Convert BP log marginals to prob
             let m_log = &bp_marginals_log[t];
             let max_v = m_log.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
             let mut m_bp: Vec<f64> = m_log.iter().map(|x| (x - max_v).exp()).collect();
             let s_bp: f64 = m_bp.iter().sum();
             for v in &mut m_bp { *v /= s_bp; }

             let diffs: Vec<f64> = (0..n_states).map(|i| (mj[i] - m_bp[i]).abs()).collect();
             println!("t={}: Exact={:?}, BP={:?}, Diff={:?}", t, mj, m_bp, diffs);

             for i in 0..n_states {
                 assert!(diffs[i] < 1e-3, "Mismatch t={} state={}", t, i);
             }
        }
    }

    #[test]
    fn test_tree_marginals() {
        // Construct a small binary tree:
        let num_vars = 7;
        let n_states = 2;

        let tree = get_test_tree(num_vars, n_states, 42); // Seed for determinism
        let exact_marginals = tree.exact_marginals();
        
        // 2. Factor Graph BP
        let mut fg = FactorGraph::new(num_vars, n_states);
        
        // Add Unary Factors from Tree
        for i in 0..num_vars {
            fg.add_factor(vec![i], tree.emissions[i].iter().map(|p| p.ln()).collect(), FactorType::Emission);
        }

        // Add Pairwise Factors from Tree
        for (idx, &(u, v)) in tree.edges.iter().enumerate() {
            fg.add_factor(vec![u, v], tree.pairwise[idx].iter().map(|p| p.ln()).collect(), FactorType::Transition);
        }

        println!("Running Belief Propagation on Tree (Nodes={})...", num_vars);
        let bp_marginals_log = fg.run_belief_propagation(50, 1e-6, 0.0);

        // 3. Compare
        println!("Comparing Marginals (Exact vs BP):");
        for i in 0..num_vars {
            let m_log = &bp_marginals_log[i];
            let max_v = m_log.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut m_bp: Vec<f64> = m_log.iter().map(|x| (x - max_v).exp()).collect();
            let s_bp: f64 = m_bp.iter().sum();
            for v in &mut m_bp { *v /= s_bp; }

            let diffs: Vec<f64> = (0..n_states).map(|s| (exact_marginals[i][s] - m_bp[s]).abs()).collect();
            println!("Node {}: Exact={:?}, BP={:?}, Diff={:?}", i, exact_marginals[i], m_bp, diffs);

            for s in 0..n_states {
                assert!(diffs[s] < 1e-4, "Mismatch node={} state={}", i, s);
            }
        }
    }

    #[test]
    fn test_hmrf_marginals() {
        // 3x3 loopy belief propagation using Potts model
        let width = 4;
        let height = 4;
        let n_states = 2; // Binary grid
        let coupling_prob: f64 = 0.8; 

        let potts = get_test_potts(width, height, n_states, coupling_prob);
        let exact_marginals = potts.exact_marginals();

        let num_vars = potts.num_vars();
        let edges = &potts.edges;

        let mut fg = FactorGraph::new(num_vars, n_states);
        
        for i in 0..num_vars {
            fg.add_factor(vec![i], potts.emissions[i].iter().map(|p| p.ln()).collect(), FactorType::Emission);
        }

        let pairwise_table = [coupling_prob, 1.0 - coupling_prob, 1.0 - coupling_prob, coupling_prob];
        let pw_log: Vec<f64> = pairwise_table.iter().map(|p| p.ln()).collect();

        // NB reference count the pairwise table since it's shared across all edges for memory efficiency.
        let pw_log_rc = Rc::new(pw_log);

        for &(u, v) in edges {
            fg.add_shared_factor(vec![u, v], pw_log_rc.clone(), FactorType::Transition);
        }

        println!("Running Loopy Belief Propagation on {}x{} grid...", width, height);
        
        // Loopy BP is approximate and iterative.
        let bp_marginals_log = fg.run_belief_propagation(100, 1e-5, 0.0);

        // 3. Compare (Expect deviations due to loops, but should be correlated)
        let mut max_diff = 0.0;
        
        for i in 0..num_vars {
            let m_log = &bp_marginals_log[i];
            let max_v = m_log.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut m_bp: Vec<f64> = m_log.iter().map(|x| (x - max_v).exp()).collect();
            let s_bp: f64 = m_bp.iter().sum();
            for v in &mut m_bp { *v /= s_bp; }

            let diffs: Vec<f64> = (0..n_states).map(|s| (exact_marginals[i][s] - m_bp[s]).abs()).collect();
            println!("Node {}: Exact={:?}, BP={:?}, Diff={:?}", i, exact_marginals[i], m_bp, diffs);
            
            for s in 0..n_states {
                if diffs[s] > max_diff { max_diff = diffs[s]; }
            }
        }

        println!("Max discrepancy in marginals: {}", max_diff);
        // Loopy BP is generally good but not exact. Assert reasonable closeness.
        assert!(max_diff < 0.05, "Loopy BP diverged significantly from exact marginals");
    }
}
