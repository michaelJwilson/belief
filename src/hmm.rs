use std::f64;

pub struct HMM {
    pub n_states: usize,
    pub trans: Vec<f64>,
    pub emit: Vec<f64>,
    pub prior: Vec<f64>,
}

impl HMM {
    pub fn new(n_states: usize, trans: Vec<f64>, emit: Vec<f64>, prior: Vec<f64>) -> Self {
        Self {
            n_states,
            trans,
            emit,
            prior,
        }
    }

    pub fn forward(&self, obs: &[usize]) -> Vec<Vec<f64>> {
        let chain_len = obs.len();
        let mut alpha = vec![vec![0.0; self.n_states]; chain_len];

        // Init alpha
        for i in 0..self.n_states {
            alpha[0][i] = self.prior[i] * self.emit[i * 2 + obs[0]];
        }
        let s0: f64 = alpha[0].iter().sum();
        for v in &mut alpha[0] { *v /= s0; }

        // Recursion
        for t in 1..chain_len {
            for j in 0..self.n_states {
                let mut p = 0.0;
                for i in 0..self.n_states {
                    p += alpha[t - 1][i] * self.trans[i * self.n_states + j];
                }
                alpha[t][j] = p * self.emit[j * 2 + obs[t]];
            }
            let st: f64 = alpha[t].iter().sum();
            for v in &mut alpha[t] { *v /= st; }
        }
        alpha
    }

    pub fn backward(&self, obs: &[usize]) -> Vec<Vec<f64>> {
        let chain_len = obs.len();
        let mut beta = vec![vec![0.0; self.n_states]; chain_len];

        // Init beta
        for i in 0..self.n_states {
            beta[chain_len - 1][i] = 1.0;
        }

        // Recursion
        for t in (0..chain_len - 1).rev() {
            for i in 0..self.n_states {
                let mut sum = 0.0;
                for j in 0..self.n_states {
                    sum += self.trans[i * self.n_states + j]
                        * self.emit[j * 2 + obs[t + 1]]
                        * beta[t + 1][j];
                }
                beta[t][i] = sum;
            }
            let sb: f64 = beta[t].iter().sum();
            for v in &mut beta[t] { *v /= sb; }
        }
        beta
    }

    pub fn marginals(&self, obs: &[usize]) -> Vec<Vec<f64>> {
        let alpha = self.forward(obs);
        let beta = self.backward(obs);
        let chain_len = obs.len();
        let mut marginals = vec![vec![0.0; self.n_states]; chain_len];

        for t in 0..chain_len {
            let mut sum = 0.0;
            for i in 0..self.n_states {
                marginals[t][i] = alpha[t][i] * beta[t][i];
                sum += marginals[t][i];
            }
            for v in &mut marginals[t] { *v /= sum; }
        }
        marginals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_marginals() {
        // NB exact inference is 2^N=32 configs for N=5.
        let n_states: usize = 2;
        let chain_len: usize = 5;

        // NB observed 2-state sequence.        
        let obs = [0, 1, 0, 1, 0];

        let hmm = HMM::new(
            n_states,
            vec![0.6, 0.4, 0.3, 0.7],
            vec![0.8, 0.2, 0.1, 0.9],
            vec![0.6, 0.4],     
        );

        let marginals = hmm.marginals(&obs);

        assert_eq!(marginals.len(), chain_len);
        for t in 0..chain_len {
            let sum: f64 = marginals[t].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}