/// Computes log(sum(exp(x))) in a numerically stable way.
pub fn logsumexp(logs: &[f64]) -> f64 {
    let max = logs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum = logs.iter().map(|&x| (x - max).exp()).sum::<f64>();
    max + sum.ln()
}
