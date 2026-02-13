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

pub fn get_test_hmm(n_states: usize, chain_len: usize) -> (HMM, Vec<usize>) {
    // Deterministic prior: favoring lower indices slightly
    let mut prior = vec![0.0; n_states];
    let mut sum = 0.0;
    for i in 0..n_states {
        prior[i] = 1.0 / (1.0 + i as f64);
        sum += prior[i];
    }
    for v in &mut prior { *v /= sum; }

    // Deterministic transitions: favoring self-loops
    let mut trans = vec![0.0; n_states * n_states];
    for i in 0..n_states {
        let mut row_sum = 0.0;
        for j in 0..n_states {
            let val = if i == j { 0.5 } else { 0.5 / (n_states as f64 - 1.0).max(1.0) };
            trans[i * n_states + j] = val;
            row_sum += val;
        }
        // Normalize just in case of float drifts, though logic above sums to ~1.0
        for j in 0..n_states { trans[i * n_states + j] /= row_sum; }
    }

    // Deterministic emissions: n_states x n_states (assuming obs size == n_states for simplicity, or 2)
    // Let's assume observation domain size is 2 for simplicity as per previous tests, 
    // or we can make it variable. The previous code assumed emission size 2 per state.
    // Let's generalize to emission vector size matching n_states * 2 (obs domain 2).
    let obs_domain = 2;
    let mut emit = vec![0.0; n_states * obs_domain];
    for i in 0..n_states {
        // State i emits 0 with prob p, 1 with prob 1-p
        let p = 0.1 + 0.8 * ((i % 2) as f64); // Alternating bias
        emit[i * obs_domain + 0] = p;
        emit[i * obs_domain + 1] = 1.0 - p;
    }

    // Deterministic observations
    let mut obs = Vec::with_capacity(chain_len);
    for t in 0..chain_len {
        obs.push(t % obs_domain);
    }

    (HMM::new(n_states, trans, emit, prior), obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_marginals() {
        // NB exact inference is 2^N=32 configs for N=5.
        let n_states: usize = 2;
        let chain_len: usize = 5;
        /*
        // NB observed 2-state sequence.        
        let obs = [0, 1, 0, 1, 0];

        let hmm = HMM::new(
            n_states,
            vec![0.6, 0.4, 0.3, 0.7],
            vec![0.8, 0.2, 0.1, 0.9],
            vec![0.6, 0.4],     
        );
        */
        let (hmm, obs) = get_test_hmm(n_states, chain_len);
        let marginals = hmm.marginals(&obs);

        assert_eq!(marginals.len(), chain_len);

        for t in 0..chain_len {
            let sum: f64 = marginals[t].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}