use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct Tree {
    pub num_vars: usize,
    pub n_states: usize,
    pub edges: Vec<(usize, usize)>,
    pub emissions: Vec<Vec<f64>>,
    pub pairwise: Vec<Vec<f64>>,
}

impl Tree {
    pub fn exact_marginals(&self) -> Vec<Vec<f64>> {
        let mut exact_marginals = vec![vec![0.0; self.n_states]; self.num_vars];
        let mut total_prob = 0.0;
        
        // NB warning: exponential complexity in num_vars
        let n_configs = self.n_states.pow(self.num_vars as u32);
        
        for config_idx in 0..n_configs {
            let mut config = vec![0; self.num_vars];
            let mut tmp = config_idx;
            for i in 0..self.num_vars {
                config[i] = tmp % self.n_states;
                tmp /= self.n_states;
            }

            // Calculate joint probability (unnormalized)
            let mut log_prob = 0.0;
            
            // Unary
            for i in 0..self.num_vars {
                log_prob += self.emissions[i][config[i]].ln();
            }

            // Pairwise
            for (edge_idx, &(u, v)) in self.edges.iter().enumerate() {
                let table_idx = config[u] * self.n_states + config[v];
                log_prob += self.pairwise[edge_idx][table_idx].ln();
            }

            let prob = log_prob.exp();
            total_prob += prob;

            for i in 0..self.num_vars {
                exact_marginals[i][config[i]] += prob;
            }
        }
        
        // Normalize
        for i in 0..self.num_vars {
            for s in 0..self.n_states {
                exact_marginals[i][s] /= total_prob;
            }
        }
        exact_marginals
    }
}

pub fn get_test_tree(num_vars: usize, n_states: usize, seed: u64) -> Tree {
    let mut rng = StdRng::seed_from_u64(seed);
    
    let mut edges = Vec::new();
    let mut frontier = vec![0];
    let mut current_vars = 1;

    // Grow tree by selecting random leaf and adding up to 2 children until num_vars reached
    while current_vars < num_vars && !frontier.is_empty() {
        let idx = rng.random_range(0..frontier.len());
        let parent = frontier.swap_remove(idx);
        
        for _ in 0..2 {
            if current_vars < num_vars {
                let child = current_vars;
                edges.push((parent, child));
                frontier.push(child);
                current_vars += 1;
            }
        }
    }

    // Identify non-leaf (internal) nodes
    let mut is_internal = vec![false; num_vars];
    for &(u, _) in &edges {
        is_internal[u] = true;
    }

    // Generate Emissions
    let mut emissions = Vec::with_capacity(num_vars);
    for i in 0..num_vars {
        if is_internal[i] {
            // Uniform emissions for internal nodes (1.0 in log domain becomes 0.0, probability 1/n)
            // Or just equal probability: 1.0 / n_states
            let p_uniform = 1.0 / n_states as f64;
            emissions.push(vec![p_uniform; n_states]);
        } else {
            // Random emissions for leaf nodes
            let mut row = vec![0.0; n_states];
            let mut sum = 0.0;
            for j in 0..n_states {
                let p: f64 = rng.random();
                row[j] = p;
                sum += p;
            }
            for j in 0..n_states { row[j] /= sum; }
            emissions.push(row);
        }
    }

    // Generate Pairwise
    let mut pairwise = Vec::with_capacity(edges.len());
    for _ in 0..edges.len() {
        let mut table = vec![0.0; n_states * n_states];
        let mut sum = 0.0;
        for j in 0..table.len() {
             let p: f64 = rng.random();
             table[j] = p;
             sum += p;
        }
        for j in 0..table.len() { table[j] /= sum; }
        pairwise.push(table);
    }
    
    Tree {
        num_vars,
        n_states,
        edges,
        emissions,
        pairwise,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_exact_marginals() {
        let num_vars = 7;
        let n_states = 2;
        
        let tree = get_test_tree(num_vars, n_states, 42); // Seed for determinism
        let exact_marginals = tree.exact_marginals();

        for i in 0..num_vars {
            let sum: f64 = exact_marginals[i].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Marginal sum is not 1.0 for node {}", i);
            for s in 0..n_states {
                assert!(exact_marginals[i][s] >= 0.0, "Marginal is negative for node {} state {}", i, s);
            }
        }
    }
}