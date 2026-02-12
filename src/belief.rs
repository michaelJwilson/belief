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
    pub table: Vec<f64>, // flattened table, row-major, probabilities for all clique assignments to this factor.
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

/// Returns a vector of marginal distributions for each variable in the factor graph.
/// Each entry is a vector of probabilities (length = variable domain size) representing
/// the estimated marginal probability of each assignment for that variable after belief propagation.
/// 
/// # Arguments
/// * `fg` - The factor graph definition.
/// * `max_iters` - Maximum number of iterations for loopy belief propagation.
/// * `tol` - Convergence tolerance for message updates.
/// * `beta` - Inverse temperature (1.0 for standard BP, Infinity for Max-Product).
pub fn ls_belief_propagation(
    fg: &FactorGraph,
    max_iters: usize,
    tol: f64,
    beta: Option<f64>,
) -> Vec<Vec<f64>> {

    let beta: f64 = beta.unwrap_or(1.0);
    let domain_size = fg.domain_size;

    println!("Solving belief propagation");

    // Check if we can use an ordered schedule based on depth
    let use_depth_schedule = fg.variables.iter().all(|v| v.depth.is_some());
    
    // Create reference to variables sorted by depth for forward/backward passes if available
    let mut sorted_vars: Vec<&Variable> = fg.variables.iter().collect();
    if use_depth_schedule {
        println!("Using depth-based schedule for message passing.");
        sorted_vars.sort_by_key(|v| v.depth.unwrap());
    }

    let mut messages: Message = HashMap::default();

    // Initialize messages from variables to factors as uniform distributions.
    for var in &fg.variables {
        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..domain_size {
                messages.insert((var.id, fid, s), 1.0 / domain_size as f64);
            }
        }
    }

    // Initialize messages from factors to variables as uniform distributions.
    for factor in &fg.factors {
        for &vid in &factor.variables {
            for s in 0..domain_size {
                messages.insert((factor.id, vid, s), 1.0 / domain_size as f64);
            }
        }
    }

    // NB converges in t*, diameter of the tree (max. node to node distance),
    //    i.e. 2log_2 num. leaves for a fully balanced (ultrametric) binary tree.
    for iter in 0..max_iters {
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
                    let mut prod = 1.0;

                    for &other_fid in fg.var_to_factors.get(&var.id).unwrap() {
                        if other_fid != fid {
                            prod *= messages[&(other_fid, var.id, s)];
                        }
                    }

                    new_messages.insert((var.id, fid, s), prod);
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
                    let mut sum = 0.0;
                    let num_vars = fvars.len();

                    let mut current_assignment = vec![0; num_vars];
                    current_assignment[i] = s;

                    loop {
                        // Compute index into factor table
                        let mut idx = 0;
                        let mut stride = 1;

                        // NB row-major: last index first.
                        for &a in current_assignment.iter().rev() {
                            idx += a * stride;
                            stride *= domain_size;
                        }

                        let mut prod = 1.0;

                        for (j, &other_vid) in fvars.iter().enumerate() {
                            if j != i {
                                prod *= messages[&(other_vid, factor.id, current_assignment[j])];
                            }
                        }

                        sum += ftable[idx].powf(beta) * prod;

                        if !next_assignment(&mut current_assignment, domain_size, i) {
                            break;
                        }
                    }

                    new_messages.insert((factor.id, vid, s), sum);
                }
            }
        }

        // NB messages are probability distributions, normalize.
        for ((from, to, _), _) in new_messages.clone().iter() {
            let norm: f64 = (0..domain_size).map(|s| new_messages[&(*from, *to, s)]).sum();

            for s in 0..domain_size {
                let val = new_messages[&(*from, *to, s)] / norm;
                new_messages.insert((*from, *to, s), val);
            }
        }
        
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
        let mut marginal = vec![1.0; domain_size];

        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..domain_size {
                marginal[s] *= messages[&(fid, var.id, s)];
            }
        }

        let norm: f64 = marginal.iter().sum();

        for s in 0..domain_size {
            marginal[s] /= norm;
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
        // Note: ls_belief_propagation normalizes messages, so it computes marginals P(X_i).
        // To get the total likelihood P(Observations), we need the normalization constant Z.
        // We simulate a message pass similar to the forward algorithm on the graph structure.

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
        let bp_marginals = ls_belief_propagation(&fg, max_iters, tol, None);

        println!("\nMarginals Comparison:");
        println!("{:>5} | {:>15} | {:>15}", "Time", "Exact (F-B)", "BP");
        
        for t in 0..chain_len {
            for i in 0..n_states {
                let exact = exact_marginals[t][i];
                let bp = bp_marginals[t][i]; // Variable ids match time index t
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
            // Table should be p (row) -> left (col)
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
        let bp_marginals = ls_belief_propagation(&fg, max_iters, tol, None);

        // 6. Compare
        println!("\nTree Marginals Comparison (Leaves and Root):");
        println!("{:>5} | {:>15} | {:>15}", "Node", "Exact (Fel)", "BP");
        
        let nodes_to_check: Vec<usize> = (0..5).chain(std::iter::once(nnodes-1)).collect(); // First 5 leaves + root

        for &id in &nodes_to_check {
             for s in 0..ncolor {
                let exact = exact_marginals[id][s];
                let bp = bp_marginals[id][s];
                let diff = (exact - bp).abs();
                println!("{:>5} | State {}: {:.6} | {:.6}", id, s, exact, bp);
                assert!(diff < 1e-5, "Mismatch at node {}, state {}: exact={}, bp={}", id, s, exact, bp);
             }
             println!("---");
        }

        // Global check
        let max_diff = exact_marginals.iter().flatten()
            .zip(bp_marginals.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        
        println!("Max difference over all nodes: {:.3e}", max_diff);
        assert!(max_diff < 1e-5, "BP did not converge to exact tree marginals");
    }
}
