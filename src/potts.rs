pub struct Potts {
    pub width: usize,
    pub height: usize,
    pub n_states: usize,
    pub coupling_prob: f64,
    pub emissions: Vec<Vec<f64>>,
    pub edges: Vec<(usize, usize)>,
}

impl Potts {
    pub fn new(
        width: usize, 
        height: usize, 
        n_states: usize, 
        coupling_prob: f64, 
        emissions: Vec<Vec<f64>>
    ) -> Self {
        let mut edges = Vec::new();
        
        // NB deterministic pairwise potentials (Potts/Ising model-like); one per edge.
        for r in 0..height {
            for c in 0..width - 1 {
                edges.push((r * width + c, r * width + c + 1));
            }
        }
        for r in 0..height - 1 {
            for c in 0..width {
                edges.push((r * width + c, (r + 1) * width + c));
            }
        }

        Self { width, height, n_states, coupling_prob, emissions, edges }
    }

    pub fn num_vars(&self) -> usize {
        self.width * self.height
    }

    pub fn exact_marginals(&self) -> Vec<Vec<f64>> {
        let num_vars = self.num_vars();
        let mut marginals = vec![vec![0.0; self.n_states]; num_vars];
        let mut total_prob = 0.0;
        
        let total_configs = self.n_states.pow(num_vars as u32);

        for config_idx in 0..total_configs {
            let mut config = vec![0; num_vars];
            let mut temp = config_idx;
            for i in 0..num_vars {
                config[i] = temp % self.n_states;
                temp /= self.n_states;
            }

            let mut log_prob = 0.0;
            
            // Unary
            for i in 0..num_vars {
                log_prob += self.emissions[i][config[i]].ln();
            }

            // Pairwise
            for &(u, v) in &self.edges {
                let su = config[u];
                let sv = config[v];
                let prob = if su == sv {
                    self.coupling_prob
                } else if self.n_states == 2 {
                    1.0 - self.coupling_prob
                } else {
                    (1.0 - self.coupling_prob) / (self.n_states as f64 - 1.0)
                };
                log_prob += prob.ln();
            }

            let prob = log_prob.exp();
            total_prob += prob;

            for i in 0..num_vars {
                marginals[i][config[i]] += prob;
            }
        }
        
        // Normalize
        for i in 0..num_vars {
            for s in 0..self.n_states {
                marginals[i][s] /= total_prob;
            }
        }
        marginals
    }
}

pub fn get_test_potts(
    width: usize, 
    height: usize, 
    n_states: usize, 
    coupling_prob: f64
) -> Potts {
    let num_vars = width * height;

    // NB deterministic emissions probability (Unary); one per site for two states.
    let mut emissions = Vec::with_capacity(num_vars);
    for i in 0..num_vars {
        let p = 0.6 + ((i as f64 * 0.1) % 0.3); // Biased towards state 0

        // NB for the variable obs. character, prob. of state 0 is p, state 1 is 1-p.
        emissions.push(vec![p, 1.0 - p]);
    }
    Potts::new(width, height, n_states, coupling_prob, emissions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_potts_exact_marginals() {
        let width = 3;
        let height = 3;
        let n_states = 2; // Binary grid
        let coupling_prob: f64 = 0.8; 
        
        let potts = get_test_potts(width, height, n_states, coupling_prob);
        let num_vars = potts.num_vars();

        let marginals = potts.exact_marginals();
        
        assert_eq!(marginals.len(), num_vars);

        for i in 0..num_vars {
            let sum: f64 = marginals[i].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}