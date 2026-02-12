use rand::prelude::*;

#[test]
fn test_random_number_generation() {
    let mut rng = rand::rng();
    let nums: Vec<i32> = (1..100).collect();
    
    let _ = nums.choose(&mut rng);
}