use std::fs::File;
use std::io::{BufWriter, Write};
use std::collections::{HashMap, VecDeque, HashSet};
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
    use rand::Rng;
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
        fg.run_belief_propagation(50, 1e-6);
        let bp_marginals_log = fg.calculate_marginals();
        
        // 4. Compare
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

             for i in 0..n_states {
                 assert!((mj[i] - m_bp[i]).abs() < 1e-3, "Mismatch t={} state={}", t, i);
             }
        }
    }

    /*
    #[test]
    fn test_tree_marginals() {
        let nleaves = 8;
        let ncolor = 2;
        let nnodes = nleaves + nleaves - 1;
        let mut rng = rand::rng();
        let mut emission_factors = Vec::new();
        for _ in 0..nleaves {
             let mut row: Vec<f64> = (0..ncolor).map(|_| rng.random::<f64>()).collect();
             let norm: f64 = row.iter().sum();
             for v in &mut row { *v /= norm; }
             emission_factors.push(row);
        }
        let mut pairwise_table = vec![0.6, 0.4, 0.3, 0.7]; // simplified

        let exact = felsensteins(nleaves, nleaves-1, ncolor, &emission_factors, &pairwise_table);
    }
    */
}
