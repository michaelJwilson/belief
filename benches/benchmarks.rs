use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use belief::belief::{FactorGraph, FactorType};

fn build_chain(n_vars: usize, n_states: usize) -> FactorGraph {
    let mut fg = FactorGraph::new(n_vars, n_states);
    let unary = vec![0.0; n_states];
    let pairwise = vec![0.0; n_states * n_states];

    for i in 0..n_vars {
        fg.add_factor(vec![i], unary.clone(), FactorType::Emission);
    }
    for i in 0..n_vars - 1 {
        fg.add_factor(vec![i, i + 1], pairwise.clone(), FactorType::Transition);
    }
    fg
}

fn build_tree(layers: usize, n_states: usize) -> FactorGraph {
    let n_vars = (1 << layers) - 1;
    let mut fg = FactorGraph::new(n_vars, n_states);
    let unary = vec![0.0; n_states];
    let pairwise = vec![0.0; n_states * n_states];

    for i in 0..n_vars {
        fg.add_factor(vec![i], unary.clone(), FactorType::Emission);
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < n_vars {
            fg.add_factor(vec![i, left], pairwise.clone(), FactorType::Transition);
        }
        if right < n_vars {
            fg.add_factor(vec![i, right], pairwise.clone(), FactorType::Transition);
        }
    }
    fg
}

fn build_hmrf(dim: usize, n_states: usize) -> FactorGraph {
    let n_vars = dim * dim;
    let mut fg = FactorGraph::new(n_vars, n_states);
    let unary = vec![0.0; n_states];
    let pairwise = vec![0.0; n_states * n_states];

    for i in 0..n_vars {
        fg.add_factor(vec![i], unary.clone(), FactorType::Emission);
    }
    for r in 0..dim {
        for c in 0..dim {
            let u = r * dim + c;
            if c + 1 < dim {
                let v = r * dim + c + 1;
                fg.add_factor(vec![u, v], pairwise.clone(), FactorType::Transition);
            }
            if r + 1 < dim {
                let v = (r + 1) * dim + c;
                fg.add_factor(vec![u, v], pairwise.clone(), FactorType::Transition);
            }
        }
    }
    fg
}

fn bench_bp(c: &mut Criterion) {
    let mut group = c.benchmark_group("BeliefPropagation");
    let n_states = 2;

    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("Chain", size), size, |b, &size| {
            b.iter_batched(
                || build_chain(size, n_states),
                |mut fg| fg.run_belief_propagation(black_box(100), black_box(1e-6)),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    for layers in [5, 7].iter() {
        let n_vars = (1 << layers) - 1;
        group.bench_with_input(BenchmarkId::new("Tree", n_vars), layers, |b, &layers| {
            b.iter_batched(
                || build_tree(layers, n_states),
                |mut fg| fg.run_belief_propagation(black_box(100), black_box(1e-6)),
                criterion::BatchSize::SmallInput,
            );
        });
    }

    for dim in [5, 10].iter() {
        let n_vars = dim * dim;
        group.bench_with_input(BenchmarkId::new("HMRF", n_vars), dim, |b, &dim| {
            b.iter_batched(
                || build_hmrf(dim, n_states),
                |mut fg| fg.run_belief_propagation(black_box(100), black_box(1e-6)),
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bp);
criterion_main!(benches);

