use rand::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Latent,
    Observed, // # TODO rename emission.
}

// # TOOD update Variable to include a depth / order parameter that, when available (for chains / trees)
// # can be used to optimize belief propagation by only passing messages forward (for chains) or from leaves to root (for trees).  
#[derive(Clone)]
pub struct Variable {
    pub id: usize,
    pub domain: usize, // # DEPRECATE varying domain; assume all variables have same domain size for simplicity.
    pub var_type: VariableType,
    pub pos: Option<(f64, f64)>, // # DEPRECATE position.
}

// NB Forney-style factor graph: factors are functions of their variables, and variables are connected to factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType {
    Emission,
    Transition,
    Start_Prior, // # TODO rename to Prior.
    Custom,
}

pub struct Factor {
    pub id: usize,
    pub variables: Vec<usize>,   // variable ids
    pub table: Vec<f64>, // flattened table, row-major, probabilities for all clique assignments to this factor.
    pub factor_type: FactorType, // label for the type of factor
}

/// NB Forney factor graph is with variables and factors connected by edges.
pub struct FactorGraph {
    pub variables: Vec<Variable>,
    pub factors: Vec<Factor>,
    pub var_to_factors: HashMap<usize, Vec<usize>>, // variable id -> factor ids.
    pub factor_to_vars: HashMap<usize, Vec<usize>>, // factor id -> variable ids.
}

// TODO add docstring - seems to assign each variable to a state in its domain.
fn next_assignment(assignment: &mut [usize], domains: &[usize], skip: usize) -> bool {
    for (j, dom) in domains.iter().enumerate() {
        if j == skip {
            continue;
        }
        assignment[j] += 1;

        if assignment[j] < *dom {
            return true;
        } else {
            assignment[j] = 0;
        }
    }

    false
}

// # TODO update to (from, to, vec over global domain).
/// Message from variable to factor or vice versa.  Indexed by (from, to, assignment).
type Message = HashMap<(usize, usize, usize), f64>;

// # TODO update docstring to include variable definition, e.g. beta and tol.
/// Returns a vector of marginal distributions for each variable in the factor graph.
/// Each entry is a vector of probabilities (length = variable domain size) representing
/// the estimated marginal probability of each assignment for that variable after belief propagation.
pub fn ls_belief_propagation(
    fg: &FactorGraph,
    max_iters: usize,
    tol: f64,
    beta: Option<f64>,
) -> Vec<Vec<f64>> {

    // # TODO complete all typing e.g. beta: f64
    let beta = beta.unwrap_or(1.0);

    println!("Solving belief propagation");

    let mut messages: Message = HashMap::default();

    // # TODO complete all variable typing
    // NB initialize var -> factor as 1/domain size.
    for var in &fg.variables {
        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..var.domain {
                messages.insert((var.id, fid, s), 1.0 / var.domain as f64);
            }
        }
    }

    for factor in &fg.factors {
        for &vid in &factor.variables {
            let vdom = fg.variables.iter().find(|v| v.id == vid).unwrap().domain;

            for s in 0..vdom {
                messages.insert((factor.id, vid, s), 1.0 / vdom as f64);
            }
        }
    }

    // NB converges in t*, diameter of the tree (max. node to node distance),
    //    i.e. 2log_2 num. leaves for a fully balanced (ultrametric) binary tree.
    for iter in 0..max_iters {
        let mut new_messages = messages.clone();

        // NB  passes on incoming messages to var (except output factor),
        //	   see eqn. (14.14) of Information, Physics & Computation, Mezard.
        for var in &fg.variables {
            for &fid in fg.var_to_factors.get(&var.id).unwrap() {
                let vdom = var.domain;

                for s in 0..vdom {
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

        // NB passes on incoming messages to factor (except output variable),
        //    weighted by factor marginalized over all other variables,
        //    see eqn. (14.15) of Information, Physics & Computation, Mezard.
        for factor in &fg.factors {
            let fvars = &factor.variables;
            let ftable = &factor.table;

            for (i, &vid) in fvars.iter().enumerate() {
                let vdom = fg.variables.iter().find(|v| v.id == vid).unwrap().domain;

                for s in 0..vdom {
                    // NB sum over all assignments to other variables in the factor
                    let mut sum = 0.0;
                    let num_vars = fvars.len();

                    let mut assignment = vec![0; num_vars];

                    let domains: Vec<usize> = fvars
                        .iter()
                        .map(|vid| fg.variables.iter().find(|v| v.id == *vid).unwrap().domain)
                        .collect();

                    assignment[i] = s;

                    loop {
                        // Compute index into factor table
                        let mut idx = 0;
                        let mut stride = 1;

                        // NB row-major: last index first.
                        for (j, &a) in assignment.iter().rev().enumerate() {
                            idx += a * stride;
                            stride *= domains[domains.len() - 1 - j];
                        }

                        let mut prod = 1.0;

                        for (j, &other_vid) in fvars.iter().enumerate() {
                            if j != i {
                                prod *= messages[&(other_vid, factor.id, assignment[j])];
                            }
                        }

                        sum += ftable[idx].powf(beta) * prod;

                        if !next_assignment(&mut assignment, &domains, i) {
                            break;
                        }
                    }

                    new_messages.insert((factor.id, vid, s), sum);
                }
            }
        }

        // NB messages are probability distributions, normalize.
        for ((from, to, _), _) in new_messages.clone().iter() {
            let vdom = if let Some(var) = fg.variables.iter().find(|v| v.id == *from) {
                var.domain
            } else {
                fg.variables.iter().find(|v| v.id == *to).unwrap().domain
            };

            let norm: f64 = (0..vdom).map(|s| new_messages[&(*from, *to, s)]).sum();

            for s in 0..vdom {
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
        let vdom = var.domain;
        let mut marginal = vec![1.0; vdom];

        for &fid in fg.var_to_factors.get(&var.id).unwrap() {
            for s in 0..vdom {
                marginal[s] *= messages[&(fid, var.id, s)];
            }
        }

        let norm: f64 = marginal.iter().sum();

        for s in 0..vdom {
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
            for child_state in 0..ncolor {
                let mut sum = 0.0;

                for parent_state in 0..ncolor {
                    let trans = pairwise_table[parent_state * ncolor + child_state];

                    // For the sibling, use the upward message
                    let sibling = if child == left { right } else { left };

                    sum += outs[p][parent_state] * trans * likelihoods[sibling][parent_state];
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

pub fn compute_tree_positions(nleaves: usize, nancestors: usize) -> Vec<(f64, f64)> {
    let nnodes = nleaves + nancestors;
    let mut pos = vec![(0.0, 0.0); nnodes];

    // Leaves: evenly spaced along x at y=0
    for i in 0..nleaves {
        pos[i] = (i as f64, 0.0);
    }

    // Ancestors: recursively place at the midpoint of their children, y increases with depth
    let mut depth = 1;
    let mut nodes_in_level = nleaves;

    // NB first parent
    let mut start_idx = nleaves;

    while start_idx < nnodes {
        let parents_in_level = nodes_in_level / 2;

        for i in 0..parents_in_level {
            let left = 2 * (start_idx + i - nleaves);
            let right = left + 1;

            let parent = start_idx + i;

            let x = (pos[left].0 + pos[right].0) / 2.0;
            let y = depth as f64;

            pos[parent] = (x, y);
        }

        nodes_in_level /= 2;

        start_idx += parents_in_level;

        depth += 1;
    }

    pos
}

pub fn save_node_marginals(
    filename: &str,
    variables: &[Variable],
    marginals: &[Vec<f64>],
    felsenstein: &[Vec<f64>],
) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "# id,x,y,bp_marginal,felsenstein_marginal")?;

    for (var, (bp, fel)) in variables
        .iter()
        .zip(marginals.iter().zip(felsenstein.iter()))
    {
        let (x, y) = var.pos.unwrap_or((f64::NAN, f64::NAN));

        writeln!(writer, "{}\t{}\t{}\t{:?}\t{:?}", var.id, x, y, bp, fel)?;
    }
    Ok(())
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
    fn test_compute_tree_positions() {
        let nleaves = 5;
        let nancestors = 4;
        let result = compute_tree_positions(nleaves, nancestors);
        assert_eq!(result.len(), nleaves + nancestors);
        for (x, y) in result {
            assert!(x >= 0.0 && y >= 0.0);
        }
    }

    #[test]
    fn test_save_node_marginals() {
        let nleaves = 5;
        let ncolor = 3;
        let variables = vec![
            Variable {
                id: 0,
                domain: ncolor,
                var_type: VariableType::Latent,
                pos: None,
            },
            Variable {
                id: 1,
                domain: ncolor,
                var_type: VariableType::Latent,
                pos: None,
            },
            Variable {
                id: 2,
                domain: ncolor,
                var_type: VariableType::Latent,
                pos: None,
            },
            Variable {
                id: 3,
                domain: ncolor,
                var_type: VariableType::Latent,
                pos: None,
            },
            Variable {
                id: 4,
                domain: ncolor,
                var_type: VariableType::Latent,
                pos: None,
            },
        ];
        let marginals = vec![
            vec![0.1, 0.3, 0.6],
            vec![0.2, 0.4, 0.4],
            vec![0.3, 0.5, 0.2],
            vec![0.4, 0.3, 0.3],
            vec![0.5, 0.2, 0.3],
        ];
        let exp = vec![
            vec![0.1, 0.3, 0.6],
            vec![0.2, 0.4, 0.4],
            vec![0.3, 0.5, 0.2],
            vec![0.4, 0.3, 0.3],
            vec![0.5, 0.2, 0.3],
        ];
        save_node_marginals("data/node_marginals.csv", &variables, &marginals, &exp).unwrap();
    }

    // TODO complete all typing., eg.g n_states, chain_len etc.
    #[test]
    fn test_hmm_likelihood() {
        let n_states = 2;
        let chain_len = 5;

        // Simple HMM parameters
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

        // Initialization
        for i in 0..n_states {
            alpha[i] = prior[i] * emit[i * 2 + obs[0]];
        }

        // # TODO update all probability manipulation to log space to avoid underflow.
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
                domain: n_states,
                var_type: VariableType::Latent,
                pos: None,
            })
            .collect();

        let mut factors = Vec::new();
        let mut var_to_factors: HashMap<usize, Vec<usize>> = HashMap::default();
        let mut factor_to_vars: HashMap<usize, Vec<usize>> = HashMap::default();

        let mut add_factor = |vars: Vec<usize>, table: Vec<f64>| {
            let fid = factors.len();
            factors.push(Factor {
                id: fid,
                variables: vars.clone(),
                table,
                factor_type: FactorType::Custom,
            });
            factor_to_vars.insert(fid, vars.clone());
            for v in vars {
                var_to_factors.entry(v).or_default().push(fid);
            }
        };

        // Prior Factor on X0
        add_factor(vec![0], prior.clone());

        // Emission Factors for X0..XT
        for (t, &o) in obs.iter().enumerate() {
            let table: Vec<f64> = (0..n_states).map(|s| emit[s * 2 + o]).collect();
            add_factor(vec![t], table);
        }

        // Transition Factors for Xt -> Xt+1
        for t in 0..chain_len - 1 {
            // Factor connects [t, t+1]. Table is flat row-major: t is row, t+1 is col.
            add_factor(vec![t, t + 1], trans.clone());
        }

        let fg = FactorGraph {
            variables,
            factors,
            var_to_factors,
            factor_to_vars,
        };

        // # TODO utilize ls_belief_propagation strictly to calculate node marginals and likelihood (approx.)
        let mut node_belief = vec![1.0; n_states]; // Implicitly message from 'left'
        
        // # TODO complete test to compare likelihood from forward pass and belief propagation

    }
}
