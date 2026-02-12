// use algos::dijkstra::{dijkstra, get_adjacencies_fixture_large};
use log::{info, warn};
// use pprof;
// use std::fs::File;

fn main() {
    /*
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1_000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();
    */

    env_logger::init();

    info!("Welcome to algos.");
    /*
    let adjs = get_adjacencies_fixture_large(100);

    match dijkstra(&adjs, 0, 50) {
        Some(cost) => {
            info!("The cost is: {}", cost);
        }
        None => {
            warn!("Failed to calculate the Dijkstra distance");
        }
    }
    */
    /*
    if let Ok(report) = guard.report().build() {
        let file = File::create("flamegraph.svg").unwrap();
        let mut options = pprof::flamegraph::Options::default();
        options.image_width = Some(2500);
        report.flamegraph_with_options(file, &mut options).unwrap();
    };
    */

    info!("Done.");
}
