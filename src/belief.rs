use std::fs::File;
use std::io::{BufWriter, Write};
use std::collections::{HashMap, VecDeque, HashSet};
use rand::Rng;
use rand::prelude::*;

use crate::utils::logsumexp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Latent,
    Emission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType {
    Emission,
    Transition,
    Prior,
    Custom,
}

#[derive(Debug, Clone)]
pub struct Factor {
    pub id: usize,
    pub factor_type: FactorType,
    pub variables: Vec<usize>,
    pub table: Vec<f64>,
}

/// NB Forney factor graph is with variables and factors connected by edges.
pub struct FactorGraph {
    pub factors: Vec<Factor>,
    pub var_adj: HashMap<usize, Vec<usize>>,
    pub domain_size: usize,
    pub num_vars: usize,
    pub var_to_factor: HashMap<(usize, usize), Vec<f64>>,
    pub factor_to_var: HashMap<(usize, usize), Vec<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeDirection {
    VarToFactor,
    FactorToVar,
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
        let id = self.factors.len();
        for &v in &variables {
            self.var_adj.entry(v).or_default().push(id);
        }
        self.factors.push(Factor {
            id,
            variables,
            table,
            factor_type,
        });
    }

    pub fn run_belief_propagation(&mut self, max_iters: usize, tolerance: f64) {
        let mut queue = VecDeque::new();

        // Initialize messages to uniform (0.0 in log domain)
        for fac in &self.factors {
            for &v in &fac.variables {
                // Initialize default keys
                self.var_to_factor.entry((v, fac.id)).or_insert_with(|| vec![0.0; self.domain_size]);
                self.factor_to_var.entry((fac.id, v)).or_insert_with(|| vec![0.0; self.domain_size]);
                
                // Add initial tasks to queue
                queue.push_back((EdgeDirection::VarToFactor, v, fac.id));
                queue.push_back((EdgeDirection::FactorToVar, fac.id, v));
            }
        }

        let mut iters = 0;
        let factor_count = self.factors.len().max(1);

        while let Some((dir, src, dst)) = queue.pop_front() {
            iters += 1;
            // Break if potentially infinite loop (loopy graph) or max iters exceeded
            if iters > max_iters * factor_count * 10 && iters > 200_000 { break; }

            match dir {
                EdgeDirection::VarToFactor => {
                    let var_id = src;
                    let factor_id = dst;
                    
                    // Sum incoming messages from all *other* factors
                    let mut incoming = vec![0.0; self.domain_size];
                    if let Some(neighbors) = self.var_adj.get(&var_id) {
                         for i in 0..self.domain_size {
                             let mut sum = 0.0;
                             for &n_fid in neighbors {
                                 if n_fid != factor_id {
                                     sum += self.factor_to_var[&(n_fid, var_id)][i];
                                 }
                             }
                             incoming[i] = sum;
                        }
                    }

                    // Normalize to control numerical stability in log domain
                    let lse = logsumexp(&incoming); 
                    for x in &mut incoming { *x -= lse; }

                    let old_msg = &self.var_to_factor[&(var_id, factor_id)];
                    let diff: f64 = incoming.iter().zip(old_msg.iter()).map(|(a, b)| (a - b).abs()).sum();
                    
                    if diff > tolerance {
                        self.var_to_factor.insert((var_id, factor_id), incoming);
                        // Trigger updates for neighbor factors
                        let target_factor = &self.factors[factor_id];
                        for &v_neighbor in &target_factor.variables {
                            if v_neighbor != var_id {
                                queue.push_back((EdgeDirection::FactorToVar, factor_id, v_neighbor));
                            }
                        }
                    }
                },
                EdgeDirection::FactorToVar => {
                    let factor_id = src;
                    let var_id = dst;
                    let factor = &self.factors[factor_id];
                    
                    if let Some(target_idx) = factor.variables.iter().position(|&x| x == var_id) {
                         let incoming_msgs: Vec<Vec<f64>> = factor.variables.iter().map(|&v| {
                            self.var_to_factor[&(v, factor_id)].clone()
                        }).collect();

                        let mut new_msg = vec![0.0; self.domain_size];
                        for state in 0..self.domain_size {
                             new_msg[state] = FactorToVarMessage::update(
                                 factor,
                                 target_idx,
                                 state,
                                 self.domain_size,
                                 &incoming_msgs
                             );
                        }
                        
                        let lse = logsumexp(&new_msg); 
                        for x in &mut new_msg { *x -= lse; }

                        let old_msg = &self.factor_to_var[&(factor_id, var_id)];
                        let diff: f64 = new_msg.iter().zip(old_msg.iter()).map(|(a, b)| (a - b).abs()).sum();

                        if diff > tolerance {
                             self.factor_to_var.insert((factor_id, var_id), new_msg);
                             // Trigger updates for neighbor variables
                             if let Some(n_fids) = self.var_adj.get(&var_id) {
                                 for &n_fid in n_fids {
                                     if n_fid != factor_id {
                                         queue.push_back((EdgeDirection::VarToFactor, var_id, n_fid));
                                     }
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
            if let Some(n_fids) = self.var_adj.get(&v) {
                for &fid in n_fids {
                    let msg = &self.factor_to_var[&(fid, v)];
                    for i in 0..self.domain_size {
                        log_prob[i] += msg[i];
                    }
                }
            }
            let lse = logsumexp(&log_prob);
            for val in &mut log_prob { *val -= lse; }
            marginals.push(log_prob);
        }
        marginals
    }
}


/// Decrements the assignment to the next valid state in the joint domain.
/// Returns true if a valid assignment was found, false if the sequence wrapped around.
fn next_assignment(assignment: &mut [usize], domain_size: usize, skip: usize) -> bool {
    for (j, val) in assignment.iter_mut().enumerate() {
        if j == skip {
            continue;
        }
        *val += 1;

        if *val < domain_size {
            return true;
        } else {
            *val = 0;
        }
    }

    false
}

pub struct VarToFactorMessage;

impl VarToFactorMessage {
    // NB simple ln. sum of incoming messages from factors to this variable;
    //    see eqn. (14.14) of Information, Physics & Computation, Mezard.
    pub fn update(
        var_id: usize, 
        target_factor_id: usize, 
        state: usize, 
        incoming_messages: &[f64]
    ) -> f64 {
        incoming_messages.iter().sum()
    }
}

pub struct FactorToVarMessage;

impl FactorToVarMessage {
    pub fn update(
        factor: &Factor,
        target_var_idx: usize,
        target_state: usize,
        domain_size: usize,
        incoming_messages: &[Vec<f64>] // Vector of messages from each variable in factor.variables
    ) -> f64 {
        // Validation: incoming_messages should correspond to Factor.variables (Var -> Factor messages)
        // assert_eq!(incoming_messages.len(), factor.variables.len());

        // NB we will sum over factor assignments bar the target variable,
        //    weighted by corresponding input variable messages.
        //    see eqn. (14.15) of Information, Physics & Computation, Mezard.
        let mut current_assignment = vec![0; factor.variables.len()];
        current_assignment[target_var_idx] = target_state;

        // NB 
        let mut log_terms = Vec::with_capacity(domain_size.pow((factor.variables.len() - 1) as u32));

        loop {
            // NB compute index into factor table
            let mut idx = 0;
            let mut stride = 1;

            // NB row-major: last index first.
            for &a in current_assignment.iter().rev() {
                idx += a * stride;
                stride *= domain_size;
            }

            let mut term = 0.0;
            
            for (j, msg) in incoming_messages.iter().enumerate() {
                if j != target_var_idx {
                    // Read message from Var (j) -> Factor
                    // msg contains the value for the specific assignment of variable j
                    term += msg[current_assignment[j]]; 
                }
            }

            // Factor potential (log) + incoming messages (log)
            log_terms.push(factor.table[idx] + term);

            if !next_assignment(&mut current_assignment, domain_size, target_var_idx) {
                break;
            }
        }

        logsumexp(&log_terms)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::seq::SliceRandom;
    use std::collections::HashMap;

    #[test]
    fn test_chain_marginals() {
        let n_states: usize = 2;
        let chain_len: usize = 5;

        let trans = vec![0.6, 0.4, 0.3, 0.7];
        let emit = vec![0.8, 0.2, 0.1, 0.9];
        let prior = vec![0.6, 0.4];
        let obs = vec![0, 1, 0, 1, 0];

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

        // Transitions
        let trans_log: Vec<f64> = trans.iter().map(|x| x.ln()).collect();
        for t in 0..chain_len-1 {
            fg.add_factor(vec![t, t+1], trans_log.clone(), FactorType::Transition);
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
        let edges = vec![
            (0, 1), (0, 2),
            (1, 3), (1, 4),
            (2, 5), (2, 6)
        ];

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
        let pairwise_table = vec![coupling_prob, 1.0 - coupling_prob, 1.0 - coupling_prob, coupling_prob];

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
        for &(u, v) in &edges {
            fg.add_factor(vec![u, v], pw_log.clone(), FactorType::Transition);
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
