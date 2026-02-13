use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use crate::utils::logsumexp;

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

/// Forney factor graph with variables and factors connected by edges.
pub struct FactorGraph {
    pub factors: Vec<Factor>,
    pub var_adj: HashMap<usize, Vec<usize>>,
    pub domain_size: usize,
    pub num_vars: usize,
    // (var_id, factor_id) -> message
    pub var_to_factor: HashMap<(usize, usize), Vec<f64>>,
    // (factor_id, var_id) -> message
    pub factor_to_var: HashMap<(usize, usize), Vec<f64>>,
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
        }
    }

    pub fn add_factor(&mut self, variables: Vec<usize>, table: Vec<f64>, factor_type: FactorType) {
        self.add_shared_factor(variables, Rc::new(table), factor_type);
    }

    pub fn add_shared_factor(&mut self, variables: Vec<usize>, table: Rc<Vec<f64>>, factor_type: FactorType) {
        let id = self.factors.len();
        for &v in &variables {
            self.var_adj.entry(v).or_default().push(id);
        }
        self.factors.push(Factor { id, variables, table, factor_type });
    }

    pub fn run_belief_propagation(&mut self, max_iters: usize, tolerance: f64) {
        let mut queue = VecDeque::new();

        // Initialize messages and populate queue
        for factor in &self.factors {
            for &var in &factor.variables {
                self.var_to_factor.entry((var, factor.id)).or_insert_with(|| vec![0.0; self.domain_size]);
                self.factor_to_var.entry((factor.id, var)).or_insert_with(|| vec![0.0; self.domain_size]);
                
                queue.push_back(WorkItem::VarToFactor { var_id: var, factor_id: factor.id });
                queue.push_back(WorkItem::FactorToVar { factor_id: factor.id, var_id: var });
            }
        }

        let mut iters = 0;
        let limit = 200_000.min(max_iters * self.factors.len().max(1) * 10).max(1000);

        while let Some(item) = queue.pop_front() {
            iters += 1;
            if iters > limit { break; }

            match item {
                WorkItem::VarToFactor { var_id, factor_id } => {
                    // Update u_{x -> f}(x) = Sum_{h in N(x)\f} u_{h -> x}(x)
                    let mut incoming = vec![0.0; self.domain_size];
                    if let Some(neighbors) = self.var_adj.get(&var_id) {
                         for i in 0..self.domain_size {
                             let mut sum = 0.0;
                             for &n_fid in neighbors {
                                 if n_fid != factor_id {
                                     // Only read existing messages
                                     if let Some(msg) = self.factor_to_var.get(&(n_fid, var_id)) {
                                         sum += msg[i];
                                     }
                                 }
                             }
                             incoming[i] = sum;
                        }
                    }
                    normalize_log_msg(&mut incoming);

                    let entry = self.var_to_factor.get_mut(&(var_id, factor_id)).unwrap();
                    let diff: f64 = incoming.iter().zip(entry.iter()).map(|(a, b)| (a - b).abs()).sum();
                    
                    if diff > tolerance {
                        *entry = incoming;

                        let factor = &self.factors[factor_id];
                        for &n_var in &factor.variables {
                            if n_var != var_id {
                                queue.push_back(WorkItem::FactorToVar { factor_id, var_id: n_var });
                            }
                        }
                    }
                }
                WorkItem::FactorToVar { factor_id, var_id } => {
                    // Update u_{f -> x}(x) = log Sum_{...} exp(...)
                    let factor = &self.factors[factor_id];
                    let target_idx = factor.variables.iter().position(|&x| x == var_id).unwrap();
                    
                    // Collect incoming Var->Factor messages
                    let incoming_msgs: Vec<Vec<f64>> = factor.variables.iter().map(|&v| {
                        self.var_to_factor.get(&(v, factor_id)).cloned().unwrap_or_else(|| vec![0.0; self.domain_size])
                    }).collect();

                    let mut new_msg = compute_factor_message(factor, target_idx, self.domain_size, &incoming_msgs);
                    normalize_log_msg(&mut new_msg);

                    let entry = self.factor_to_var.get_mut(&(factor_id, var_id)).unwrap();
                    let diff: f64 = new_msg.iter().zip(entry.iter()).map(|(a, b)| (a - b).abs()).sum();

                    if diff > tolerance {
                        *entry = new_msg;
                        // Schedule neighbors
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
    }

    pub fn calculate_marginals(&self) -> Vec<Vec<f64>> {
        let mut marginals = Vec::with_capacity(self.num_vars);
        for v in 0..self.num_vars {
            let mut log_prob = vec![0.0; self.domain_size];
            // Sum all incoming Factor->Var messages
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
}

fn normalize_log_msg(msg: &mut [f64]) {
    let lse = logsumexp(msg);
    for x in msg { *x -= lse; }
}

fn compute_factor_message(
    factor: &Factor,
    target_var_idx: usize, // Index in factor.variables
    domain_size: usize,
    incoming_messages: &[Vec<f64>]
) -> Vec<f64> {
    let mut messages = vec![0.0; domain_size];
    
    // We iterate over the state of the target variable (the one we are sending TO)
    for target_state in 0..domain_size {
        let mut current_assignment = vec![0; factor.variables.len()];
        current_assignment[target_var_idx] = target_state;
        
        let mut log_terms = Vec::new();

        // Sum over all other variables' configurations
        loop {
            // Compute table index (row-major)
            let mut idx = 0;
            let mut stride = 1;
            for &a in current_assignment.iter().rev() {
                idx += a * stride;
                stride *= domain_size;
            }

            let mut term = factor.table[idx];
            for (j, msg) in incoming_messages.iter().enumerate() {
                if j != target_var_idx {
                    term += msg[current_assignment[j]]; 
                }
            }
            log_terms.push(term);

            if !next_assignment(&mut current_assignment, domain_size, target_var_idx) {
                break;
            }
        }

        messages[target_state] = logsumexp(&log_terms);
    }
    messages
}

fn next_assignment(assignment: &mut [usize], domain_size: usize, skip: usize) -> bool {
    for (j, val) in assignment.iter_mut().enumerate() {
        if j == skip { continue; }
        *val += 1;
        if *val < domain_size { return true; }
        *val = 0;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use rand::seq::SliceRandom;
    use std::collections::HashMap;

    #[test]
    fn test_chain_marginals() {
        let n_states: usize = 2;
        let chain_len: usize = 5;

        let trans = [0.6, 0.4, 0.3, 0.7];
        let emit = [0.8, 0.2, 0.1, 0.9];
        let prior = [0.6, 0.4];
        let obs = [0, 1, 0, 1, 0];

        // 1. Standard Forward-Backward Calculation for Validation
        let mut alpha = vec![vec![0.0; n_states]; chain_len];
        // Init alpha
        for i in 0..n_states { alpha[0][i] = prior[i] * emit[i * 2 + obs[0]]; }
        let s0: f64 = alpha[0].iter().sum();
        for v in &mut alpha[0] { *v /= s0; }

        for t in 1..chain_len {
            for j in 0..n_states {
                let mut p = 0.0;
                for i in 0..n_states { p += alpha[t-1][i] * trans[i * n_states + j]; }
                alpha[t][j] = p * emit[j * 2 + obs[t]];
            }
            let st: f64 = alpha[t].iter().sum();
            for v in &mut alpha[t] { *v /= st; }
        }

        let mut beta = vec![vec![0.0; n_states]; chain_len];
        for i in 0..n_states { beta[chain_len-1][i] = 1.0; }

        for t in (0..chain_len-1).rev() {
            for i in 0..n_states {
                let mut sum = 0.0;
                for j in 0..n_states {
                    sum += trans[i * n_states + j] * emit[j * 2 + obs[t+1]] * beta[t+1][j];
                }
                beta[t][i] = sum;
            }
            let sb: f64 = beta[t].iter().sum();
            for v in &mut beta[t] { *v /= sb; }
        }
        
        // 2. Build Factor Graph
        let mut fg = FactorGraph::new(chain_len, n_states);
        
        // Prior
        fg.add_factor(vec![0], prior.iter().map(|x| x.ln()).collect(), FactorType::Prior);

        // Transitions (Shared)
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
    
        // 3. Run BP
        println!("Running Belief Propagation on Chain (L={})...", chain_len);
        fg.run_belief_propagation(50, 1e-6);
        let bp_marginals_log = fg.calculate_marginals();
        
        // 4. Compare
        println!("Comparing Marginals (Exact vs BP):");
        assert_eq!(bp_marginals_log.len(), chain_len);
        
        for t in 0..chain_len {
             // Convert exact components to marginals
             let mut mj = vec![0.0; n_states];
             let mut mj_sum = 0.0;
             for i in 0..n_states {
                 mj[i] = alpha[t][i] * beta[t][i];
                 mj_sum += mj[i];
             }
             for i in 0..n_states { mj[i] /= mj_sum; }

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
        //        0
        //      /   \
        //     1     2
        //    / \   / \
        //   3   4 5   6
        
        let num_vars = 7;
        let n_states = 2;
        let edges = [(0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5), (2, 6)];

        // Deterministic emissions (unary potentials)
        let mut emissions = Vec::with_capacity(num_vars);
        for i in 0..num_vars {
            // Generate deterministic probabilities based on index
            let p = 0.1 + ((i as f64 * 0.13) % 0.8);
            emissions.push(vec![p, 1.0 - p]);
        }

        // Deterministic transitions (pairwise potentials) - simple symmetric for edges
        let mut pairwise = Vec::with_capacity(edges.len());
        for i in 0..edges.len() {
             let p_stay = 0.6 + ((i as f64 * 0.05) % 0.3); // Values between 0.6 and 0.9
             pairwise.push(vec![p_stay, 1.0 - p_stay, 1.0 - p_stay, p_stay]);
        }

        // 1. Brute Force Exact Inference
        // Since N=7, states=2, total configs = 2^7 = 128. Feasible.
        let mut exact_marginals = vec![vec![0.0; n_states]; num_vars];
        let mut total_prob = 0.0;

        for config_idx in 0..(1 << num_vars) {
            let mut config = vec![0; num_vars];
            for i in 0..num_vars {
                if (config_idx >> i) & 1 == 1 {
                    config[i] = 1;
                }
            }

            // Calculate joint probability (unnormalized)
            let mut log_prob = 0.0;
            
            // Unary
            for i in 0..num_vars {
                log_prob += emissions[i][config[i]].ln();
            }

            // Pairwise
            for (edge_idx, &(u, v)) in edges.iter().enumerate() {
                let table_idx = config[u] * n_states + config[v];
                log_prob += pairwise[edge_idx][table_idx].ln();
            }

            let prob = log_prob.exp();
            total_prob += prob;

            for i in 0..num_vars {
                exact_marginals[i][config[i]] += prob;
            }
        }

        // Normalize exact marginals
        for i in 0..num_vars {
            for s in 0..n_states {
                exact_marginals[i][s] /= total_prob;
            }
        }

        // 2. Factor Graph BP
        let mut fg = FactorGraph::new(num_vars, n_states);
        
        // Add Unary Factors
        for i in 0..num_vars {
            fg.add_factor(vec![i], emissions[i].iter().map(|p| p.ln()).collect(), FactorType::Emission);
        }

        // Add Pairwise Factors
        for (edge_idx, &(u, v)) in edges.iter().enumerate() {
            fg.add_factor(vec![u, v], pairwise[edge_idx].iter().map(|p| p.ln()).collect(), FactorType::Transition);
        }

        println!("Running Belief Propagation on Tree (Nodes={})...", num_vars);
        fg.run_belief_propagation(50, 1e-6);
        let bp_marginals_log = fg.calculate_marginals();

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
        // Construct a small 3x3 Grid (Loopy Graph)
        // 0 -- 1 -- 2
        // |    |    |
        // 3 -- 4 -- 5
        // |    |    |
        // 6 -- 7 -- 8
        
        let width = 3;
        let height = 3;
        let num_vars = width * height;
        let n_states = 2; // Binary grid

        let mut edges = Vec::new();
        // Horizontal edges
        for r in 0..height {
            for c in 0..width - 1 {
                let u = r * width + c;
                let v = r * width + c + 1;
                edges.push((u, v));
            }
        }
        // Vertical edges
        for r in 0..height - 1 {
            for c in 0..width {
                let u = r * width + c;
                let v = (r + 1) * width + c;
                edges.push((u, v));
            }
        }

        // Deterministic emissions (Unary)
        let mut emissions = Vec::with_capacity(num_vars);
        for i in 0..num_vars {
            let p = 0.6 + ((i as f64 * 0.1) % 0.3); // Biased towards state 0
            emissions.push(vec![p, 1.0 - p]);
        }

        // Deterministic pairwise potentials (Potts/Ising model-like)
        // Strong coupling to make loops relevant
        let coupling_prob: f64 = 0.8; 
        let pairwise_table = [coupling_prob, 1.0 - coupling_prob, 1.0 - coupling_prob, coupling_prob];

        // 1. Brute Force Exact Inference
        // 2^9 = 512 total configurations. Slightly larger but instant.
        let mut exact_marginals = vec![vec![0.0; n_states]; num_vars];
        let mut total_prob = 0.0;

        for config_idx in 0..(1 << num_vars) {
            let mut config = vec![0; num_vars];
            for i in 0..num_vars {
                if (config_idx >> i) & 1 == 1 {
                    config[i] = 1;
                }
            }

            let mut log_prob = 0.0;
            // Unary
            for i in 0..num_vars {
                log_prob += emissions[i][config[i]].ln();
            }
            // Pairwise
            for &(u, v) in &edges {
                let table_idx = config[u] * n_states + config[v];
                log_prob += pairwise_table[table_idx].ln();
            }

            let prob = log_prob.exp();
            total_prob += prob;

            for i in 0..num_vars {
                exact_marginals[i][config[i]] += prob;
            }
        }
        
        // Normalize
        for i in 0..num_vars {
            for s in 0..n_states {
                exact_marginals[i][s] /= total_prob;
            }
        }

        // 2. Build Factor Graph / Run Loopy BP
        let mut fg = FactorGraph::new(num_vars, n_states);
        
        for i in 0..num_vars {
            fg.add_factor(vec![i], emissions[i].iter().map(|p| p.ln()).collect(), FactorType::Emission);
        }

        let pw_log: Vec<f64> = pairwise_table.iter().map(|p| p.ln()).collect();
        let pw_log_rc = Rc::new(pw_log);

        for &(u, v) in &edges {
            fg.add_shared_factor(vec![u, v], pw_log_rc.clone(), FactorType::Transition);
        }

        println!("Running Loopy Belief Propagation on 3x3 Grid...");
        // Loopy BP is approximate and iterative.
        fg.run_belief_propagation(100, 1e-5);
        let bp_marginals_log = fg.calculate_marginals();

        // 3. Compare (Expect deviations due to loops, but should be correlated)
        println!("Comparing Marginals (Exact vs Loopy BP):");
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
