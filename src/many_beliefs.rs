use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use crate::utils::logsumexp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType { Prior, Emission, Transition }

// NB Thread-safe sharing of the table with a Atomic Reference Count.
#[derive(Debug, Clone)]
pub struct Factor {
    pub id: usize,
    pub factor_type: FactorType,
    pub variables: Vec<usize>,
    pub table: Arc<Vec<f64>>, 
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WorkItem {
    VarToFactor { var_id: usize, factor_id: usize },
    FactorToVar { factor_id: usize, var_id: usize },
}

// NB RwLock allows multiple threads to read simultaneously,
// but ensures exclusive access to write.
struct GraphState {
    var_outgoing: Vec<RwLock<HashMap<usize, Vec<f64>>>>, // vec[var_id] = {factor_id: message for domain}
    factor_outgoing: Vec<RwLock<HashMap<usize, Vec<f64>>>>, // vec[factor_id] = {var_id: message for domain}
}

// NB Thread-safe sharing of the state (messages) with a Atomic Reference Count.
pub struct AsyncFactorGraph {
    pub factors: Vec<Factor>,
    pub var_adj: Vec<Vec<usize>>, 
    pub domain_size: usize,
    pub num_vars: usize,
    state: Arc<GraphState>,
}

impl AsyncFactorGraph {
    pub fn new(num_vars: usize, domain_size: usize) -> Self {
        let mut var_adj = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            var_adj.push(Vec::new());
        }

        // NB read-write locks for outgoing messages from variable
        let mut var_outgoing = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            var_outgoing.push(RwLock::new(HashMap::new()));
        }

        Self {
            factors: Vec::new(),
            var_adj,
            domain_size,
            num_vars,
            state: Arc::new(GraphState {
                var_outgoing,
                factor_outgoing: Vec::new(),
            }),
        }
    }

    pub fn add_factor(&mut self, variables: Vec<usize>, table: Vec<f64>, factor_type: FactorType) {
        self.add_shared_factor(variables, Arc::new(table), factor_type);
    }
    
    pub fn add_shared_factor(&mut self, variables: Vec<usize>, table: Arc<Vec<f64>>, factor_type: FactorType) {
        let id = self.factors.len();
        
        // NB update var_adj with this factor id and initialize outgoing variable message for this factor.
        for &v in &variables {
            self.var_adj[v].push(id);
            self.state.var_outgoing[v]
                .write()
                .unwrap()
                .insert(id, vec![0.0; self.domain_size]);
        }

        let mut f_msgs = HashMap::new();
        for &v in &variables {
            f_msgs.insert(v, vec![0.0; self.domain_size]);
        }
        
        // NB push handles factor id assignment, with map handling variable id assignment.
        if let Some(state_mut) = Arc::get_mut(&mut self.state) {
            state_mut.factor_outgoing.push(RwLock::new(f_msgs));
        } else {
            panic!("Cannot add factors to AsyncFactorGraph after initial declaration.");
        }

        self.factors.push(Factor { id, variables, table, factor_type });
    }

    pub fn calculate_marginals(&self) -> Vec<Vec<f64>> {
        let mut marginals = Vec::with_capacity(self.num_vars);
        for v in 0..self.num_vars {
            let mut log_prob = vec![0.0; self.domain_size];
            for &fid in &self.var_adj[v] {
                if let Some(lock) = self.state.factor_outgoing.get(fid) {
                    let map = lock.read().unwrap();
                    if let Some(msg) = map.get(&v) {
                        for k in 0..self.domain_size {
                            log_prob[k] += msg[k];
                        }
                    }
                }
            }
            normalize_log_msg(&mut log_prob);
            marginals.push(log_prob);
        }
        marginals
    }

    pub fn run_belief_propagation(
        &self, 
        max_iters: usize, 
        tolerance: f64, 
        alpha: f64
    ) -> Vec<Vec<f64>> {
        let mut current_queue: Vec<WorkItem> = Vec::new();

        for factor in &self.factors {
            match factor.factor_type {
                FactorType::Prior | FactorType::Emission => {
                    for &var in &factor.variables {
                        current_queue.push(WorkItem::FactorToVar { factor_id: factor.id, var_id: var });
                    }
                },
                _ => {}
            }
        }

        let mut iters = 0;
        
        while !current_queue.is_empty() && iters < max_iters {
            // Process current batch in parallel and collect new tasks
            // We use fold/reduce to merge vectors of new work efficiently
            let next_queue: Vec<WorkItem> = current_queue.par_iter()
                .fold(Vec::new, |mut acc, item| {
                    match item {
                        WorkItem::VarToFactor { var_id, factor_id } => {
                            process_var_to_factor(self, *var_id, *factor_id, tolerance, alpha, &mut acc);
                        }
                        WorkItem::FactorToVar { factor_id, var_id } => {
                            process_factor_to_var(self, *factor_id, *var_id, tolerance, alpha, &mut acc);
                        }
                    }
                    acc
                })
                .flatten()
                .collect();
            
            // Deduplicate to avoid redundant work (optional but good for performance stability)
            // For simplicity in this snippet, we just swap. In production, a HashSet dedupe is often used.
            current_queue = next_queue;
            iters += 1;
        }
        
        if iters >= max_iters {
            println!("Rayon BP termined at max iterations.");
        }

        self.calculate_marginals()
    }
}

fn process_var_to_factor(
    graph: &AsyncFactorGraph,
    var_id: usize,
    factor_id: usize,
    tolerance: f64,
    alpha: f64,
    queue: &mut Vec<WorkItem>
) {
    let domain_size = graph.domain_size;
    let mut total_sum = vec![0.0; domain_size];
    let mut inputs_count = 0;

    let neighbors = &graph.var_adj[var_id];
    for &n_fid in neighbors {
        if let Some(lock) = graph.state.factor_outgoing.get(n_fid) {
            let map = lock.read().unwrap();
            if let Some(msg) = map.get(&var_id) {
                for i in 0..domain_size {
                    total_sum[i] += msg[i];
                }
                inputs_count += 1;
            }
        }
    }

    let mut incoming = vec![0.0; domain_size];
    if inputs_count > 0 {
        let mut target_val = vec![0.0; domain_size];
        let mut found_target = false;
        
        if let Some(lock) = graph.state.factor_outgoing.get(factor_id) {
             let map = lock.read().unwrap();
             if let Some(msg) = map.get(&var_id) {
                 target_val = msg.clone();
                 found_target = true;
             }
        }

        if found_target {
            for i in 0..domain_size {
                incoming[i] = total_sum[i] - target_val[i];
            }
        } else {
            incoming = total_sum;
        }
    }

    normalize_log_msg(&mut incoming);

    let mut update_needed = false;
    {
        let mut map = graph.state.var_outgoing[var_id].write().unwrap();
        let entry = map.entry(factor_id).or_insert(vec![0.0; domain_size]);

        if alpha < 1.0 {
            for i in 0..domain_size {
                incoming[i] = alpha * incoming[i] + (1.0 - alpha) * entry[i];
            }
            normalize_log_msg(&mut incoming);
        }

        let diff: f64 = incoming.iter().zip(entry.iter()).map(|(a, b)| (a - b).abs()).sum();
        if diff > tolerance {
            *entry = incoming;
            update_needed = true;
        }
    }

    if update_needed {
        let factor = &graph.factors[factor_id];
        for &n_var in &factor.variables {
            if n_var != var_id {
                queue.push(WorkItem::FactorToVar { factor_id, var_id: n_var });
            }
        }
    }
}

fn process_factor_to_var(
    graph: &AsyncFactorGraph,
    factor_id: usize,
    var_id: usize,
    tolerance: f64,
    alpha: f64,
    queue: &mut Vec<WorkItem>
) {
    let factor = &graph.factors[factor_id];
    let domain_size = graph.domain_size;
    let target_idx = factor.variables.iter().position(|&x| x == var_id).unwrap();

    let incoming_msgs: Vec<Vec<f64>> = factor.variables.iter().map(|&v| {
        let mut msg = vec![0.0; domain_size]; 
        if let Some(lock) = graph.state.var_outgoing.get(v) {
            let map = lock.read().unwrap();
            if let Some(m) = map.get(&factor_id) {
                msg = m.clone();
            }
        }
        msg
    }).collect();

    let mut new_msg = compute_factor_message(factor, target_idx, domain_size, &incoming_msgs);
    normalize_log_msg(&mut new_msg);

    let mut update_needed = false;
    {
        let mut map = graph.state.factor_outgoing[factor_id].write().unwrap();
        let entry = map.entry(var_id).or_insert(vec![0.0; domain_size]);

        if alpha < 1.0 {
            for i in 0..domain_size {
                new_msg[i] = alpha * new_msg[i] + (1.0 - alpha) * entry[i];
            }
            normalize_log_msg(&mut new_msg);
        }

        let diff: f64 = new_msg.iter().zip(entry.iter()).map(|(a, b)| (a - b).abs()).sum();
        if diff > tolerance {
            *entry = new_msg;
            update_needed = true;
        }
    }

    if update_needed {
        let neighbors = &graph.var_adj[var_id];
        for &n_fid in neighbors {
            if n_fid != factor_id {
                queue.push(WorkItem::VarToFactor { var_id, factor_id: n_fid });
            }
        }
    }
}

fn normalize_log_msg(msg: &mut [f64]) {
    let lse = logsumexp(msg);
    for x in msg { *x -= lse; }
}

fn compute_factor_message(
    factor: &Factor,
    target_var_idx: usize,
    domain_size: usize,
    incoming_messages: &[Vec<f64>]
) -> Vec<f64> {
    let mut messages = vec![0.0; domain_size];
    let mut strides = vec![1; factor.variables.len()];
    for i in (0..factor.variables.len()-1).rev() {
        strides[i] = strides[i+1] * domain_size;
    }
    for target_state in 0..domain_size {
        let mut current_assignment = vec![0; factor.variables.len()];
        current_assignment[target_var_idx] = target_state;
        let mut log_terms = Vec::new();
        loop {
            let mut idx = 0;
            for (k, &val) in current_assignment.iter().enumerate() {
                idx += val * strides[k];
            }
            let mut term = factor.table[idx];
            for (j, msg) in incoming_messages.iter().enumerate() {
                if j != target_var_idx {
                    term += msg[current_assignment[j]];
                }
            }
            log_terms.push(term);
            if !next_assignment_skipping(&mut current_assignment, domain_size, target_var_idx) {
                break;
            }
        }
        messages[target_state] = logsumexp(&log_terms);
    }
    messages
}

fn next_assignment_skipping(assignment: &mut [usize], domain_size: usize, skip: usize) -> bool {
    for (j, val) in assignment.iter_mut().enumerate().rev() {
        if j == skip { continue; }
        *val += 1;
        if *val < domain_size { return true; }
        *val = 0;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::hmm::get_test_hmm;
    use crate::potts::get_test_potts;
    use crate::tree::get_test_tree;

    #[test]
    fn test_chain_marginals() {
        let n_states: usize = 3;
        let chain_len: usize = 10;
        let alpha = 1.0;

        println!("Running with {} Rayon worker threads", rayon::current_num_threads());

        let (hmm, obs) = get_test_hmm(n_states, chain_len);
        let (prior, trans, emit) = (hmm.prior.clone(), hmm.trans.clone(), hmm.emit.clone());

        let exact_marginals = hmm.marginals(&obs);

        let mut fg = AsyncFactorGraph::new(chain_len, n_states);
        fg.add_factor(vec![0], prior.iter().map(|x| x.ln()).collect(), FactorType::Prior);

        let trans_log: Vec<f64> = trans.iter().map(|x| x.ln()).collect();
        let trans_log_arc = Arc::new(trans_log);
        for t in 0..chain_len-1 {
            fg.add_shared_factor(vec![t, t+1], trans_log_arc.clone(), FactorType::Transition);
        }
        for t in 0..chain_len {
            let mut emit_log = vec![0.0; n_states];
            for k in 0..n_states {
                emit_log[k] = emit[k * 2 + obs[t]].ln(); 
            }
            fg.add_factor(vec![t], emit_log, FactorType::Emission);
        }

        println!("Running belief propagation on {}-chain", chain_len);

        let bp_marginals_log = fg.run_belief_propagation(500, 1e-6, alpha);

        assert_eq!(bp_marginals_log.len(), chain_len);

        for t in 0..chain_len {
             // NB marginals from Forward-Backward via HMM struct.
             let mj = &exact_marginals[t];

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
        let num_vars = 7;
        let n_states = 2;
        let tree = get_test_tree(num_vars, n_states, 42); 
        let exact_marginals = tree.exact_marginals();
        
        println!("Running with {} Rayon worker threads", rayon::current_num_threads());

        let mut fg = AsyncFactorGraph::new(num_vars, n_states);
        let alpha = 1.0;
        for i in 0..num_vars {
            fg.add_factor(vec![i], tree.emissions[i].iter().map(|p| p.ln()).collect(), FactorType::Emission);
        }
        for (idx, &(u, v)) in tree.edges.iter().enumerate() {
            fg.add_factor(vec![u, v], tree.pairwise[idx].iter().map(|p| p.ln()).collect(), FactorType::Transition);
        }
        println!("Running on belief propagation on {}-Tree", num_vars);
        let bp_marginals_log = fg.run_belief_propagation(500, 1e-6, alpha);

        for i in 0..num_vars {
            let m_log = &bp_marginals_log[i];
            let max_v = m_log.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mut m_bp: Vec<f64> = m_log.iter().map(|x| (x - max_v).exp()).collect();
            let s_bp: f64 = m_bp.iter().sum();
            for v in &mut m_bp { *v /= s_bp; }
            let diffs: Vec<f64> = (0..n_states).map(|s| (exact_marginals[i][s] - m_bp[s]).abs()).collect();
            for s in 0..n_states {
                assert!(diffs[s] < 1e-4, "Mismatch node={} state={}", i, s);
            }
        }
    }

    #[test]
    fn test_hmrf_marginals() {
        let width = 4;
        let height = 4;
        let n_states = 2; 
        let coupling_prob: f64 = 0.8; 
        
        println!("Running with {} Rayon worker threads", rayon::current_num_threads());

        let potts = get_test_potts(width, height, n_states, coupling_prob);
        let exact_marginals = potts.exact_marginals();
        let num_vars = potts.num_vars();
        let edges = &potts.edges;
        let mut fg = AsyncFactorGraph::new(num_vars, n_states);
        let alpha = 1.0;
        for i in 0..num_vars {
            fg.add_factor(vec![i], potts.emissions[i].iter().map(|p| p.ln()).collect(), FactorType::Emission);
        }
        let pairwise_table = [coupling_prob, 1.0 - coupling_prob, 1.0 - coupling_prob, coupling_prob];
        let pw_log: Vec<f64> = pairwise_table.iter().map(|p| p.ln()).collect();
        let pw_log_arc = Arc::new(pw_log);
        for &(u, v) in edges {
            fg.add_shared_factor(vec![u, v], pw_log_arc.clone(), FactorType::Transition);
        }
        println!("Running belief propagation on {}x{} hmrf...", width, height);
        let bp_marginals_log = fg.run_belief_propagation(10_000, 1e-5, alpha);

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
        assert!(max_diff < 0.05, "Loopy BP diverged significantly from exact marginals");
    }

    #[test]
    fn test_large_hmrf_marginals() {
        let num_workers = 4;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .unwrap();

        pool.install(|| {
            let width = 1_000;
            let height = 1_000;
            let n_states = 4;
            let coupling_prob: f64 = 0.8; 

            println!("Running with {} Rayon worker threads", rayon::current_num_threads());

            // NB alpha is the damping factor for message updates. 1.0 means no damping, 0.0 means full damping (retain old).
            let alpha = 1.0;

            let potts = get_test_potts(width, height, n_states, coupling_prob);
            let num_vars = potts.num_vars();

            let edges = &potts.edges;

            let mut fg = AsyncFactorGraph::new(num_vars, n_states);

            for i in 0..num_vars {
                fg.add_factor(vec![i], potts.emissions[i].iter().map(|p| p.ln()).collect(), FactorType::Emission);
            }

            let mut pairwise_table = Vec::with_capacity(n_states * n_states);
            let off_diag = (1.0 - coupling_prob) / (n_states as f64 - 1.0);
            for i in 0..n_states {
                for j in 0..n_states {
                    if i == j {
                        pairwise_table.push(coupling_prob);
                    } else {
                        pairwise_table.push(off_diag);
                    }
                }
            }
            let pw_log: Vec<f64> = pairwise_table.iter().map(|p| p.ln()).collect();

            // NB reference count the pairwise table since it's shared across all edges for memory efficiency.
            let pw_log_arc = Arc::new(pw_log);

            for &(u, v) in edges {
                fg.add_shared_factor(vec![u, v], pw_log_arc.clone(), FactorType::Transition);
            }

            println!("Running belief propagation on {}x{} hmrf", width, height);
            
            let bp_marginals_log = fg.run_belief_propagation(50, 1e-5, alpha);
            
            assert_eq!(bp_marginals_log.len(), num_vars);
        });
    }
}
