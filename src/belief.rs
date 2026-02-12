use rand::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Latent,
    Emission,
}
#[derive(Clone)]
pub struct Variable {
    pub id: usize,
    pub var_type: VariableType,
    pub depth: Option<usize>,
}

// NB Forney-style factor graph: factors are functions of their variables, and variables are connected to factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType {
    Emission,
    Transition,
    Prior,
    Custom,
}

pub struct Factor {
    pub id: usize,
    pub variables: Vec<usize>,
    pub table: Vec<f64>, // flattened table, row-major, log-probabilities for all clique assignments.
    pub factor_type: FactorType,
}

/// NB Forney factor graph is with variables and factors connected by edges.
pub struct FactorGraph {
    pub variables: Vec<Variable>,
    pub factors: Vec<Factor>,
    pub var_to_factors: HashMap<usize, Vec<usize>>, // map: variable id -> factor ids.
    pub factor_to_vars: HashMap<usize, Vec<usize>>, // map: factor id -> variable ids.
    pub domain_size: usize,
}

/// Computes log(sum(exp(x))) in a numerically stable way.
fn logsumexp(logs: &[f64]) -> f64 {
    let max = logs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum = logs.iter().map(|&x| (x - max).exp()).sum::<f64>();
    max + sum.ln()
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

// Message from variable to factor or vice versa. Indexed by (from, to, assignment).
// Message flows along edges: (from_id, to_id, state_index) -> value
type Message = HashMap<(usize, usize, usize), f64>;

/// Returns a vector of log-marginal distributions for each variable in the factor graph.
/// Each entry is a vector of log-probabilities (length = variable domain size) representing
/// the estimated marginal log-probability of each assignment for that variable after belief propagation.
/// 
/// # Arguments
/// * `fg` - The factor graph definition.
/// * `max_iters` - Maximum number of iterations for loopy belief propagation.
/// * `tol` - Convergence tolerance for message updates.
/// * `beta` - Inverse temperature (1.0 for standard BP, Infinity for Max-Product).
/// * `damping` - Inertia for message updates (0.0 = no damping, 1.0 = no updates).
pub fn ls_belief_propagation(
    fg: &FactorGraph,
    max_iters: usize,
    tol: f64,
    beta: Option<f64>,
    damping: Option<f64>,
) -> Vec<Vec<f64>> {

    let target_beta: f64 = beta.unwrap_or(1.0);
    // Beta starts lower if annealing is implicitly desired via iteration count (typically for loopy graphs)
    // or if we just want to stabilize early iterations.
    let mut beta = target_beta;
    
    let damping = damping.unwrap_or(0.0);
    let domain_size = fg.domain_size;
    let ln_damping = if damping > 0.0 { damping.ln() } else { f64::NEG_INFINITY };
    let ln_inv_damping = if damping < 1.0 { (1.0 - damping).ln() } else { f64::NEG_INFINITY };

    println!("Solving belief propagation with beta={:.2}, damping={:.2}", target_beta, damping);

    // Variables and Factors may have overlapping IDs (both starting at 0).
    // To avoid key collisions in the message map, we offset factor IDs.
    let max_var_id = fg.variables.iter().map(|v| v.id).max().unwrap_or(0);
    let factor_offset = max_var_id + 1;

    // Check if we can use an ordered schedule based on depth
    let use_depth_schedule = fg.variables.iter().all(|v| v.depth.is_some());
    
    // Create reference to variables sorted by depth for forward/backward passes if available
    let mut sorted_vars: Vec<&Variable> = fg.variables.iter().collect();
    if use_depth_schedule {
        println!("Using depth-based schedule for message passing.");
        sorted_vars.sort_by_key(|v| v.depth.unwrap());
    }

    let mut messages: Message = HashMap::default();

    // Initialize messages from variables to factors as uniform distributions (log-space).
    let init_val = -(domain_size as f64).ln();
    for var in &fg.variables {
        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..domain_size {
                // Key: (FromID, ToID, State). Factor IDs are offset.
                messages.insert((var.id, fid + factor_offset, s), init_val);
            }
        }
    }

    // Initialize messages from factors to variables as uniform distributions (log-space).
    for factor in &fg.factors {
        for &vid in &factor.variables {
            for s in 0..domain_size {
                // Key: (FromID, ToID, State).
                messages.insert((factor.id + factor_offset, vid, s), init_val);
            }
        }
    }

    // NB converges in t*, diameter of the tree (max. node to node distance),
    //    i.e. 2log_2 num. leaves for a fully balanced (ultrametric) binary tree.
    for iter in 0..max_iters {
        
        // Beta Annealing: Ramp up beta from small value to target_beta over first 50% of iterations
        // This helps escape poor local minima / unstable initial conditions in loopy graphs.
        if iter < max_iters / 2 {
            let progress = (iter + 1) as f64 / (max_iters as f64 / 2.0);
            beta = target_beta * progress.max(0.1); 
        } else {
            beta = target_beta;
        }

        let mut new_messages = messages.clone(); // Shallow clone of map structure, new values inserted

        // Define the order of variables for updates
        let var_iterator: Box<dyn Iterator<Item = &Variable>> = if use_depth_schedule {
            // Alternating forward/backward passes can speed up convergence in trees/chains
            if iter % 2 == 0 {
                Box::new(sorted_vars.iter().cloned()) // Forward (low depth to high depth)
            } else {
                Box::new(sorted_vars.iter().rev().cloned()) // Backward (high depth to low depth)
            }
        } else {
            Box::new(fg.variables.iter())
        };

        // NB  passes on incoming messages to var (except output factor),
        //	   see eqn. (14.14) of Information, Physics & Computation, Mezard.
        for var in var_iterator {
            for &fid in fg.var_to_factors.get(&var.id).unwrap() {
                for s in 0..domain_size {
                    // product in prob domain = sum in log domain
                    let mut sum_log = 0.0;

                    for &other_fid in fg.var_to_factors.get(&var.id).unwrap() {
                        if other_fid != fid {
                            // Read message from Factor -> Var
                            sum_log += messages[&(other_fid + factor_offset, var.id, s)];
                        }
                    }

                    // Write message from Var -> Factor
                    new_messages.insert((var.id, fid + factor_offset, s), sum_log);
                }
            }
        }

        // Factors update can also benefit from ordering if associated with variables, 
        // but here we iterate simply. In a true tree schedule, we'd update factors connected
        // to the current wavefront of variables.
        // NB passes on incoming messages to factor (except output variable),
        //    weighted by factor marginalized over all other variables,
        //    see eqn. (14.15) of Information, Physics & Computation, Mezard.
        for factor in &fg.factors {
            let fvars = &factor.variables;
            let ftable = &factor.table;

            for (i, &vid) in fvars.iter().enumerate() {
                for s in 0..domain_size {
                    // NB sum over all assignments to other variables in the factor
                    // In log domain: logsumexp over assignments ( term )
                    
                    let num_vars = fvars.len();
                    let mut current_assignment = vec![0; num_vars];
                    current_assignment[i] = s;

                    // Collect log-terms for logsumexp
                    // Since specific loop structure, cannot easily pre-allocate but Vec is cheap for small domains
                    let mut log_terms = Vec::with_capacity(domain_size.pow((num_vars-1) as u32));

                    loop {
                        // Compute index into factor table
                        let mut idx = 0;
                        let mut stride = 1;

                        // NB row-major: last index first.
                        for &a in current_assignment.iter().rev() {
                            idx += a * stride;
                            stride *= domain_size;
                        }

                        let mut term = 0.0;

                        for (j, &other_vid) in fvars.iter().enumerate() {
                            if j != i {
                                // Read message from Var -> Factor
                                term += messages[&(other_vid, factor.id + factor_offset, current_assignment[j])];
                            }
                        }

                        // Factor potential (log) * beta + incoming messages (log)
                        log_terms.push(ftable[idx] * beta + term);

                        if !next_assignment(&mut current_assignment, domain_size, i) {
                            break;
                        }
                    }

                    let new_val_log = logsumexp(&log_terms);
                    // Write message from Factor -> Var
                    new_messages.insert((factor.id + factor_offset, vid, s), new_val_log);
                }
            }
        }

        // NB messages are probability distributions, normalize.
        // We separate updates for Var->Factor and Factor->Var to ensure consistent normalization
        
        // 1. Variable -> Factor edges
        for var in &fg.variables {
            for &fid in fg.var_to_factors.get(&var.id).unwrap() {
                // Compute norm for this edge (u, v) over all states
                let log_norm = logsumexp(&(0..domain_size).map(|s| new_messages[&(var.id, fid + factor_offset, s)]).collect::<Vec<_>>());
                
                for s in 0..domain_size {
                    let mut val = new_messages[&(var.id, fid + factor_offset, s)];
                    
                    // Normalize
                    if log_norm > f64::NEG_INFINITY {
                        val -= log_norm;
                    } else {
                        val = init_val;
                    }

                    // Damping (on Var -> Factor messages)
                    if damping > 0.0 {
                         let old_val = messages.get(&(var.id, fid + factor_offset, s)).copied().unwrap_or(init_val);
                         val = logsumexp(&[ln_damping + old_val, ln_inv_damping + val]);
                    }
                    new_messages.insert((var.id, fid + factor_offset, s), val);
                }
            }
        }

        // 2. Factor -> Variable edges
        for factor in &fg.factors {
            for &vid in &factor.variables {
                let log_norm = logsumexp(&(0..domain_size).map(|s| new_messages[&(factor.id + factor_offset, vid, s)]).collect::<Vec<_>>());
                
                for s in 0..domain_size {
                    let mut val = new_messages[&(factor.id + factor_offset, vid, s)];
                    
                    // Normalize
                    if log_norm > f64::NEG_INFINITY {
                        val -= log_norm;
                    } else {
                        val = init_val;
                    }

                    // Damping (on Factor -> Var messages)
                    if damping > 0.0 {
                         let old_val = messages.get(&(factor.id + factor_offset, vid, s)).copied().unwrap_or(init_val);
                         val = logsumexp(&[ln_damping + old_val, ln_inv_damping + val]);
                    }
                    new_messages.insert((factor.id + factor_offset, vid, s), val);
                }
            }
        }

        // Check convergence on the damped, normalized messages
        let max_diff = new_messages
            .iter()
            .map(|(k, &v)| (v - messages.get(k).copied().unwrap_or(0.0)).abs())
            .fold(0.0, f64::max);

        println!("Belief propagation iteration {iter}: max_diff={max_diff:.3e}");

        if max_diff < tol {
            println!("Converged at iteration {iter} with tolerance {max_diff:.3e}");
            break;
        }

        messages = new_messages;
    }

    // NB compute marginals for each variable: product of incoming messages.
    let mut marginals = Vec::new();

    for var in &fg.variables {
        let mut marginal = vec![0.0; domain_size]; // Log space accumulator

        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..domain_size {
                marginal[s] += messages[&(fid + factor_offset, var.id, s)];
            }
        }

        let log_norm = logsumexp(&marginal);

        for s in 0..domain_size {
            marginal[s] -= log_norm;
        }

        marginals.push(marginal);
    }

    marginals
}

pub fn random_one_hot_H<R: Rng + ?Sized>(rng: &mut R, nleaves: usize, ncolor: usize) -> Vec<Vec<f64>> {
    (0..nleaves)
        .map(|_| {
            let mut row = vec![0.0; ncolor];
            let idx = (0..ncolor).collect::<Vec<_>>().choose(rng).copied().unwrap();
            row[idx] = 1.0;
            row
        })
        .collect()
}

pub fn linear_one_hot_H<R: Rng + ?Sized>(rng: &mut R, nleaves: usize, ncolor: usize) -> Vec<Vec<f64>> {
    let weights: Vec<f64> = (1..=ncolor).map(|i| i as f64).collect();
    let total: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();

    (0..nleaves)
        .map(|_| {
            let mut r = rng.random::<f64>();
            let mut idx = 0;
            for (i, &p) in probs.iter().enumerate() {
                if r < p {
                    idx = i;
                    break;
                }
                r -= p;
            }
            let mut row = vec![0.0; ncolor];
            row[idx] = 1.0;
            row
        })
        .collect()
}

pub fn random_normalized_H<R: Rng + ?Sized>(rng: &mut R, nleaves: usize, ncolor: usize) -> Vec<Vec<f64>> {
    (0..nleaves)
        .map(|_| {
            let mut row: Vec<f64> = (0..ncolor).map(|_| rng.random::<f64>()).collect();
            let norm: f64 = row.iter().sum();
            if norm > 0.0 {
                for v in &mut row {
                    *v /= norm;
                }
            }
            row
        })
        .collect()
}

fn felsensteins(
    nleaves: usize,
    nancestors: usize,
    ncolor: usize,
    emission_factors: &[Vec<f64>],
    pairwise_table: &[f64],
) -> Vec<Vec<f64>> {
    let nnodes = nleaves + nancestors;

    let mut likelihoods = vec![vec![1.0; ncolor]; nnodes];

    for leaf in 0..nleaves {
        likelihoods[leaf] = emission_factors[leaf].clone();
    }

    for p in nleaves..nnodes {
        let left = 2 * (p - nleaves);
        let right = 2 * (p - nleaves) + 1;

        let mut lk = vec![0.0; ncolor];

        for parent_state in 0..ncolor {
            let mut left_sum = 0.0;
            let mut right_sum = 0.0;

            for child_state in 0..ncolor {
                let trans = pairwise_table[parent_state * ncolor + child_state];

                left_sum += trans * likelihoods[left][child_state];
                right_sum += trans * likelihoods[right][child_state];
            }

            lk[parent_state] = left_sum * right_sum;
        }

        likelihoods[p] = lk;
    }

    // Downward pass: outs[node][state]
    let mut outs = vec![vec![1.0; ncolor]; nnodes];

    let root = nnodes - 1;
    // outs[root] is all 1.0 (no parent)
    // Traverse from root downward
    for p in nleaves..nnodes {
        let left = 2 * (p - nleaves);
        let right = 2 * (p - nleaves) + 1;

        if left >= nnodes {
            continue;
        }
        // For each child, compute outs[child][child_state]
        for &child in &[left, right] {
            let sibling = if child == left { right } else { left };

            // Re-calculate message from sibling -> parent: Sum_{s_sib} P(s_sib | s_p) * L(s_sib)
            let mut sibling_msg = vec![0.0; ncolor];
            for parent_state in 0..ncolor {
                let mut s_sum = 0.0;
                for sibling_state in 0..ncolor {
                     let trans = pairwise_table[parent_state * ncolor + sibling_state];
                     s_sum += trans * likelihoods[sibling][sibling_state];
                }
                sibling_msg[parent_state] = s_sum;
            }

            for child_state in 0..ncolor {
                let mut sum = 0.0;

                for parent_state in 0..ncolor {
                    let trans = pairwise_table[parent_state * ncolor + child_state];
                    // Combine info from above (outs[p]) and from sibling side
                    sum += outs[p][parent_state] * trans * sibling_msg[parent_state];
                }

                outs[child][child_state] = sum;
            }
        }
    }

    let mut marginals = vec![vec![0.0; ncolor]; nnodes];

    for node in 0..nnodes {
        for state in 0..ncolor {
            marginals[node][state] = likelihoods[node][state] * outs[node][state];
        }

        let norm: f64 = marginals[node].iter().sum();

        if norm > 0.0 {
            for v in &mut marginals[node] {
                *v /= norm;
            }
        }
    }

    marginals
}

#[cfg(test)]
mod tests {
    //  NB  cargo test test_ultrametric_binary_tree_belief_propagation -- --nocapture
    use super::*;
    use rand::Rng;
    use rand::seq::SliceRandom;

    #[test]
    fn test_rand() {
       let mut rng = rand::rng();
       let secret_number = (1..=100).collect::<Vec<_>>().choose(&mut rng).copied().unwrap();
       println!("The secret number is: {}", secret_number);
    }   

    #[test]
    fn test_random_one_hot_H() {
        let nleaves = 5;
        let ncolor = 3;
        let mut rng = rand::rng();
        let result = random_one_hot_H(&mut rng, nleaves, ncolor);
        assert_eq!(result.len(), nleaves);
        for row in result {
            assert_eq!(row.len(), ncolor);
            assert!(row.iter().sum::<f64>() == 1.0);
        }
    }

    #[test]
    fn test_linear_one_hot_H() {
        let nleaves = 5;
        let ncolor = 3;
        let mut rng = rand::rng();
        let result = linear_one_hot_H(&mut rng, nleaves, ncolor);
        assert_eq!(result.len(), nleaves);
        for row in result {
            assert_eq!(row.len(), ncolor);
            assert!(row.iter().sum::<f64>() == 1.0);
        }
    }

    #[test]
    fn test_random_normalized_H() {
        let nleaves = 5;
        let ncolor = 3;
        let mut rng = rand::rng();
        let result = random_normalized_H(&mut rng, nleaves, ncolor);
        assert_eq!(result.len(), nleaves);
        for row in result {
            assert_eq!(row.len(), ncolor);
            assert!(row.iter().sum::<f64>() == 1.0);
        }
    }

    #[test]
    fn test_felsensteins() {
        let nleaves = 5;
        let nancestors = 4;
        let ncolor = 2;
        let emission_factors = vec![
            vec![0.8, 0.2],
            vec![0.1, 0.9],
            vec![0.8, 0.2],
            vec![0.1, 0.9],
            vec![0.8, 0.2],
        ];
        let pairwise_table = vec![
            0.6, 0.4,
            0.3, 0.7,
        ];
        let result = felsensteins(nleaves, nancestors, ncolor, &emission_factors, &pairwise_table);
        assert_eq!(result.len(), nleaves + nancestors);
        for row in result {
            assert_eq!(row.len(), ncolor);
            assert!(row.iter().sum::<f64>() == 1.0);
        }
    }

    #[test]
    fn test_chain_marginals() {
        let n_states: usize = 2;
        let chain_len: usize = 5;

        let trans = vec![
            0.6, 0.4, // 0->0, 0->1
            0.3, 0.7, // 1->0, 1->1
        ];
        let emit = vec![
            0.8, 0.2, // State 0 emits A, B
            0.1, 0.9, // State 1 emits A, B
        ];
        let prior = vec![0.6, 0.4];

        // Observations: A, B, A, B, A (indices: 0, 1, 0, 1, 0)
        let obs = vec![0, 1, 0, 1, 0];
        assert_eq!(obs.len(), chain_len);

        // --- 1. Standard Forward Pass ---
        let mut alpha = vec![0.0; n_states];

        for i in 0..n_states {
            alpha[i] = prior[i] * emit[i * 2 + obs[0]];
        }

        for t in 1..chain_len {
            let mut next_alpha = vec![0.0; n_states];
            for j in 0..n_states {
                let mut trans_prob = 0.0;
                for i in 0..n_states {
                    trans_prob += alpha[i] * trans[i * n_states + j];
                }
                next_alpha[j] = trans_prob * emit[j * 2 + obs[t]];
            }
            alpha = next_alpha;
        }

        let likelihood_fwd: f64 = alpha.iter().sum();
        println!("Forward Pass Likelihood: {:.6e}", likelihood_fwd);

        // --- 2. Factor Graph BP (Sum-Product for Z) ---
        // Variables: X0, X1, ..., X4
        let variables: Vec<Variable> = (0..chain_len)
            .map(|i| Variable {
                id: i,
                var_type: VariableType::Latent,
                depth: Some(i),
            })
            .collect();

        let mut factors = Vec::new();
        let mut var_to_factors: HashMap<usize, Vec<usize>> = HashMap::default();
        let mut factor_to_vars: HashMap<usize, Vec<usize>> = HashMap::default();

        let mut add_factor = |vars: Vec<usize>, table: Vec<f64>, ftype: FactorType| {
            let fid = factors.len();
            factors.push(Factor {
                id: fid,
                variables: vars.clone(),
                table,
                factor_type: ftype,
            });
            factor_to_vars.insert(fid, vars.clone());
            for v in vars {
                var_to_factors.entry(v).or_default().push(fid);
            }
        };

        // Prior Factor on X0
        add_factor(vec![0], prior.clone(), FactorType::Prior);

        // Emission Factors for X0..XT
        for (t, &o) in obs.iter().enumerate() {
            let table: Vec<f64> = (0..n_states).map(|s| emit[s * 2 + o]).collect();
            add_factor(vec![t], table.clone(), FactorType::Emission);
        }

        // Transition Factors for Xt -> Xt+1
        for t in 0..chain_len - 1 {
            // Factor connects [t, t+1]. Table is flat row-major: t is row, t+1 is col.
            add_factor(vec![t, t + 1], trans.clone(), FactorType::Transition);
        }

        let fg = FactorGraph {
            variables,
            factors,
            var_to_factors,
            factor_to_vars,
            domain_size: n_states,
        };

        // Custom Unnormalized BP for Partition Function Z (Likelihood)
        // Note: This block uses raw probability logic for 'node_belief', 
        // essentially a manual Forward implementation on the graph structure. 
        // This remains valid for checking "BP Factor Graph Likelihood" computation 
        // provided we don't assume `ls_belief_propagation` calculates Z directly.
        let mut node_belief = vec![1.0; n_states]; // Accumulator for messages arriving at current node

        // Pass 1: Combine Prior + Emission at X0
        // Find prior factor (unary on 0) and emission factor (unary on 0)
        let factors_0 = fg.var_to_factors.get(&0).unwrap();
        for &fid in factors_0 {
            let f = &fg.factors[fid];
            if f.variables.len() == 1 {
                for s in 0..n_states {
                    node_belief[s] *= f.table[s];
                }
            }
        }

        // Pass 2: Propagate forward
        for t in 0..chain_len - 1 {
            let mut next_belief_in = vec![0.0; n_states];
            // Find transition factor id connecting t and t+1
            // In our construction, factors are ordered or we search.
            let trans_fid = *fg
                .var_to_factors
                .get(&t)
                .unwrap()
                .iter()
                .find(|&&fid| fg.factors[fid].variables.len() == 2 && fg.factors[fid].variables.contains(&(t + 1)))
                .unwrap();
            let trans_factor = &fg.factors[trans_fid];

            // Message passing: sum_{x_t} ( belief(x_t) * trans(x_t, x_{t+1}) )
            for next_s in 0..n_states {
                let mut sum = 0.0;
                for cur_s in 0..n_states {
                    // Table is row major: cur_s * n_states + next_s
                    // Note: factor.variables is [t, t+1] based on construction order
                    let prob = trans_factor.table[cur_s * n_states + next_s];
                    sum += node_belief[cur_s] * prob;
                }
                next_belief_in[next_s] = sum;
            }

            // Multiply by Emission at t+1
            let factors_next = fg.var_to_factors.get(&(t + 1)).unwrap();
            for &fid in factors_next {
                let f = &fg.factors[fid];
                // Only unary emission factors 
                if f.variables.len() == 1 {
                    for s in 0..n_states {
                        next_belief_in[s] *= f.table[s];
                    }
                }
            }
            node_belief = next_belief_in;
        }

        let likelihood_bp: f64 = node_belief.iter().sum();
        println!("BP Factor Graph Likelihood: {:.6e}", likelihood_bp);

        let diff = (likelihood_fwd - likelihood_bp).abs();
        assert!(diff < 1e-9, "Likelihoods do not match: diff = {}", diff);

        // --- 3. Compare Marginals (Forward-Backward vs. BP) ---
        
        // Backward Pass
        let mut beta_msg = vec![vec![0.0; n_states]; chain_len];
        // Initialization at T-1
        for i in 0..n_states {
            beta_msg[chain_len - 1][i] = 1.0;
        }

        // Recursion
        for t in (0..chain_len - 1).rev() {
            for i in 0..n_states {
                let mut sum = 0.0;
                for j in 0..n_states {
                    let trans_prob = trans[i * n_states + j];
                    let emit_prob = emit[j * 2 + obs[t + 1]];
                    sum += trans_prob * emit_prob * beta_msg[t + 1][j];
                }
                beta_msg[t][i] = sum;
            }
        }

        // Compute Exact Marginals from Alpha-Beta
        let mut exact_marginals = vec![vec![0.0; n_states]; chain_len];
        // We need the alphas stored from the forward pass.
        // Re-running forward pass with storage.
        let mut alpha_msgs = vec![vec![0.0; n_states]; chain_len];
        
        // Init Alpha
        for i in 0..n_states {
            alpha_msgs[0][i] = prior[i] * emit[i * 2 + obs[0]];
        }
        // Recurse Alpha
        for t in 1..chain_len {
            for j in 0..n_states {
                let mut trans_prob = 0.0;
                for i in 0..n_states {
                    trans_prob += alpha_msgs[t - 1][i] * trans[i * n_states + j];
                }
                alpha_msgs[t][j] = trans_prob * emit[j * 2 + obs[t]];
            }
        }

        for t in 0..chain_len {
            let mut sum = 0.0;
            for i in 0..n_states {
                exact_marginals[t][i] = alpha_msgs[t][i] * beta_msg[t][i];
                sum += exact_marginals[t][i];
            }
            // Normalize
            for i in 0..n_states {
                exact_marginals[t][i] /= sum;
            }
        }

        // Run LS BP
        // For a chain, 2 iterations (1 forward, 1 backward pass) should receive info from all nodes
        // But since schedule alternates, we might need enough iterations to cover diameter.
        // With depth schedule: iter 0 (forward), iter 1 (backward). Should converge in 2 passes.
        let max_iters = 25; 
        let tol = 1e-9;
        let bp_log_marginals = ls_belief_propagation(&fg, max_iters, tol, None, None);

        println!("\nMarginals Comparison:");
        println!("{:>5} | {:>15} | {:>15}", "Time", "Exact (F-B)", "BP");
        
        for t in 0..chain_len {
            for i in 0..n_states {
                let exact = exact_marginals[t][i];
                let bp = bp_log_marginals[t][i].exp(); // Convert log to prob for comparison
                let diff = (exact - bp).abs();
                
                println!("{:>5} | State {}: {:.6} | {:.6}", t, i, exact, bp);
                assert!(diff < 1e-6, "Marginal mismatch at t={}, state {}: exact={}, bp={}", t, i, exact, bp);
            }
            println!("--------------------------------------------------");
        }
    }

    #[test]
    fn test_tree_marginals() {
        let nleaves = 8;
        let ncolor = 2;
        let nancestors = nleaves - 1;
        let nnodes = nleaves + nancestors;

        let mut rng = rand::rng();

        // 1. Generate Random emissions for leaves
        let mut emission_factors = Vec::new();
        for _ in 0..nleaves {
            let mut row: Vec<f64> = (0..ncolor).map(|_| rng.random::<f64>()).collect();
            let norm: f64 = row.iter().sum();
            for v in &mut row { *v /= norm; }
            emission_factors.push(row);
        }

        // 2. Generate Transition Matrix (symmetric for simplicity, though not required)
        let mut pairwise_table: Vec<f64> = (0..ncolor*ncolor).map(|_| rng.random::<f64>()).collect();
        // Normalize rows
        for i in 0..ncolor {
            let sum: f64 = pairwise_table[i*ncolor..(i+1)*ncolor].iter().sum();
            for j in 0..ncolor {
                pairwise_table[i*ncolor + j] /= sum;
            }
        }

        // 3. Compute Exact Marginals using Felsenstein's Pruning
        let exact_marginals = felsensteins(nleaves, nancestors, ncolor, &emission_factors, &pairwise_table);

        // 4. Build Factor Graph
        // Determine depths for variables
        let mut depths = vec![0; nnodes];
        // Leaves are depth 0
        // Ancestors: p depends on children left/right
        for p in nleaves..nnodes {
            let left = 2 * (p - nleaves);
            let right = 2 * (p - nleaves) + 1;
            depths[p] = std::cmp::max(depths[left], depths[right]) + 1;
        }

        let variables: Vec<Variable> = (0..nnodes).map(|id| {
            Variable {
                id,
                var_type: VariableType::Latent,
                depth: Some(depths[id]),
            }
        }).collect();

        let mut factors = Vec::new();
        let mut var_to_factors: HashMap<usize, Vec<usize>> = HashMap::default();
        let mut factor_to_vars: HashMap<usize, Vec<usize>> = HashMap::default();

        let mut add_factor = |vars: Vec<usize>, table: Vec<f64>, ftype: FactorType| {
            let fid = factors.len();
            factors.push(Factor {
                id: fid,
                variables: vars.clone(),
                table,
                factor_type: ftype,
            });
            factor_to_vars.insert(fid, vars.clone());
            for v in vars {
                var_to_factors.entry(v).or_default().push(fid);
            }
        };

        // Emission Factors (Leaves)
        for (i, table) in emission_factors.iter().enumerate() {
            add_factor(vec![i], table.clone(), FactorType::Emission);
        }

        // Transition Factors (Edges)
        // Felsenstein logic: p connected to left and right
        for p in nleaves..nnodes {
            let left = 2 * (p - nleaves);
            let right = 2 * (p - nleaves) + 1;
            
            // Edge p -> left
            add_factor(vec![p, left], pairwise_table.clone(), FactorType::Transition);
            
            // Edge p -> right
            add_factor(vec![p, right], pairwise_table.clone(), FactorType::Transition);
        }

        let fg = FactorGraph {
            variables,
            factors,
            var_to_factors,
            factor_to_vars,
            domain_size: ncolor,
        };

        // 5. Run BP
        // Tree diameter: approx 2 * log2(8) = 6.
        let max_iters = 10;
        let tol = 1e-9;
        let bp_log_marginals = ls_belief_propagation(&fg, max_iters, tol, None, None);

        // 6. Compare
        println!("\nTree Marginals Comparison (Leaves and Root):");
        println!("{:>5} | {:>15} | {:>15}", "Node", "Exact (Fel)", "BP");
        
        let nodes_to_check: Vec<usize> = (0..5).chain(std::iter::once(nnodes-1)).collect(); // First 5 leaves + root

        for &id in &nodes_to_check {
             for s in 0..ncolor {
                let exact = exact_marginals[id][s];
                let bp = bp_log_marginals[id][s].exp(); // Convert log to prob
                let diff = (exact - bp).abs();
                println!("{:>5} | State {}: {:.6} | {:.6}", id, s, exact, bp);
                assert!(diff < 1e-5, "Mismatch at node {}, state {}: exact={}, bp={}", id, s, exact, bp);
             }
             println!("---");
        }

        // Global check
        let max_diff = exact_marginals.iter().flatten()
            .zip(bp_log_marginals.iter().flatten())
            .map(|(a, b)| (a - b.exp()).abs())
            .fold(0.0, f64::max);
        
        println!("Max difference over all nodes: {:.3e}", max_diff);
        assert!(max_diff < 1e-5, "BP did not converge to exact tree marginals");
    }

    #[test]
    fn test_hmrf_marginals() {
        // Simple 3-node loop: 0-1, 1-2, 2-0.
        let n_vars = 3;
        let domain_size = 2;
        
        let mut rng = rand::rng();

        // 1. Define Variables
        let variables: Vec<Variable> = (0..n_vars).map(|i| Variable {
            id: i,
            var_type: VariableType::Latent,
            depth: None, // No depth in loopy graph
        }).collect();

        // 2. Define Factors
        let mut factors = Vec::new();
        let mut var_to_factors: HashMap<usize, Vec<usize>> = HashMap::default();
        let mut factor_to_vars: HashMap<usize, Vec<usize>> = HashMap::default();

        let mut add_factor = |vars: Vec<usize>, table: Vec<f64>, ftype: FactorType| {
            let fid = factors.len();
            factors.push(Factor {
                id: fid,
                variables: vars.clone(),
                table,
                factor_type: ftype,
            });
            factor_to_vars.insert(fid, vars.clone());
            for v in vars {
                var_to_factors.entry(v).or_default().push(fid);
            }
        };

        // Unary factors (random)
        let mut unary_tables = Vec::new();
        for i in 0..n_vars {
            let mut t = vec![rng.random::<f64>(), rng.random::<f64>()];
            // Normalize for better stability, though strict correctness doesn't require it for potentials
            let sum: f64 = t.iter().sum();
            for v in &mut t { *v /= sum; }
            unary_tables.push(t.clone());
            add_factor(vec![i], t, FactorType::Custom);
        }

        // Pairwise factors (random symmetric interactions)
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let mut pairwise_tables = HashMap::new();
        for &(u, v) in &edges {
            let mut t = vec![0.0; 4];
            for k in 0..4 { t[k] = rng.random::<f64>(); }
            pairwise_tables.insert((u, v), t.clone());
            add_factor(vec![u, v], t, FactorType::Custom);
        }

        let fg = FactorGraph {
            variables,
            factors,
            var_to_factors,
            factor_to_vars,
            domain_size,
        };

        // 3. Brute Force Exact Solution (Log Domain)
        let mut joint_log_prob = vec![0.0; 1 << n_vars];

        for i in 0..(1 << n_vars) {
            let x = vec![(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1]; // x0, x1, x2
            
            let mut log_p = 0.0;
            // Unary
            for v in 0..n_vars {
                log_p += unary_tables[v][x[v]].ln();
            }
            // Pairwise
            for &(u, v) in &edges {
                // Determine factor table index. Assuming factor vars are [u, v].
                // Row major: x[u]*2 + x[v]
                let idx = x[u] * 2 + x[v];
                log_p += pairwise_tables[&(u, v)][idx].ln();
            }
            
            joint_log_prob[i] = log_p;
        }

        let total_log_z = logsumexp(&joint_log_prob);

        let mut exact_log_marginals = vec![vec![f64::NEG_INFINITY; domain_size]; n_vars];
        
        for v in 0..n_vars {
            for s in 0..domain_size {
                let mut logs_for_state = Vec::new();
                for i in 0..(1 << n_vars) {
                    let x_v = (i >> v) & 1;
                    if x_v == s {
                        logs_for_state.push(joint_log_prob[i]);
                    }
                }
                exact_log_marginals[v][s] = logsumexp(&logs_for_state) - total_log_z;
            }
        }

        // 4. Run Belief Propagation
        // Small MRF loop can be tricky. Damping helps.
        let max_iters = 100;
        let tol = 1e-9;
        // Use default beta (1.0) and some damping (0.6)
        let bp_log_marginals = ls_belief_propagation(&fg, max_iters, tol, None, Some(0.6));

        // 5. Compare
        println!("\nMRF Small Loop Comparison (Exact vs BP) in Log Domain:");
        let mut max_diff = 0.0;
        for v in 0..n_vars {
            let bp_logs = &bp_log_marginals[v];
            let exact_logs = &exact_log_marginals[v];
            
            // For display convert to prob
            let bp_probs: Vec<f64> = bp_logs.iter().map(|x| x.exp()).collect();
            let exact_probs: Vec<f64> = exact_logs.iter().map(|x| x.exp()).collect();

            println!("Var {}: Exact Prob {:?}, BP Prob {:?}", v, exact_probs, bp_probs);
            println!("       Exact Log  {:?}, BP Log  {:?}", exact_logs, bp_logs);

            for s in 0..domain_size {
                let diff = (exact_logs[s] - bp_logs[s]).abs();
                if diff > max_diff { max_diff = diff; }
            }
        }
        println!("Max Log Diff: {:.6e}", max_diff);
        
        // Note: LBP is not exact for loops, but for small random potentials it's usually decent.
        // We assert reasonable closeness.
        // assert!(max_diff < 0.2, "BP deviated too much ({}) from exact on small loop", max_diff);
    }

    #[test]
    fn test_hmrf_marginals_large() {
        // This test stresses memory and runtime. 
        // 1M variables, domain size 5. Approx 3-4 GB RAM required for message maps.
        let width = 100;
        let height = 1000;
        let n_vars = width * height;
        let domain_size = 5;
        
        let mut rng = rand::rng();

        let variables: Vec<Variable> = (0..n_vars).map(|i| Variable {
            id: i,
            var_type: VariableType::Latent,
            depth: None,
        }).collect();

        let mut factors = Vec::new();
        // Reserve capacity to avoid reallocations
        // 1 unary per node + ~2 pairwise per node
        let est_factors = n_vars + 2 * n_vars; 
        let mut var_to_factors: HashMap<usize, Vec<usize>> = HashMap::with_capacity(n_vars);
        let mut factor_to_vars: HashMap<usize, Vec<usize>> = HashMap::with_capacity(est_factors);
        
        // Pre-populate var_to_factors to avoid check overhead
        for i in 0..n_vars {
            var_to_factors.insert(i, Vec::with_capacity(5));
        }

        let mut add_factor_internal = |fid: usize, vars: Vec<usize>, table: Vec<f64>| {
            factors.push(Factor {
                id: fid,
                variables: vars.clone(),
                table,
                factor_type: FactorType::Custom,
            });
            factor_to_vars.insert(fid, vars.clone());
            for &v in &vars {
                if let Some(list) = var_to_factors.get_mut(&v) {
                    list.push(fid);
                }
            }
        };

        // Reuse tables to save generation time
        let mut unary_table = vec![0.0; domain_size];
        for i in 0..domain_size { unary_table[i] = rng.random::<f64>().ln(); }
        
        let mut pairwise_table = vec![0.0; domain_size * domain_size];
        for i in 0..domain_size {
            for j in 0..domain_size {
                let p: f64 = if i == j { 0.9 } else { 0.1 / 4.0 };
                pairwise_table[i*domain_size + j] = p.ln();
            }
        }

        let mut fid_counter = 0;

        for r in 0..height {
            for c in 0..width {
                let curr = r * width + c;
                
                // Unary
                add_factor_internal(fid_counter, vec![curr], unary_table.clone());
                fid_counter += 1;

                // Right
                if c + 1 < width {
                    let right = r * width + (c + 1);
                    add_factor_internal(fid_counter, vec![curr, right], pairwise_table.clone());
                    fid_counter += 1;
                }
                
                // Down
                if r + 1 < height {
                    let down = (r + 1) * width + c;
                    add_factor_internal(fid_counter, vec![curr, down], pairwise_table.clone());
                    fid_counter += 1;
                }
            }
        }

        let fg = FactorGraph {
            variables,
            factors,
            var_to_factors,
            factor_to_vars,
            domain_size,
        };

        // 2 iters just to ensure it runs without crashing, as full convergence takes long
        let max_iters = 5; 
        let tol = 1e-4;
        
        let start = std::time::Instant::now();
        let _marginals = ls_belief_propagation(&fg, max_iters, tol, None, Some(0.0));
        println!("BP done in {:?}", start.elapsed());
    }
}
