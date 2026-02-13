//! Comprehensive Strength Testing Binary
//!
//! Runs extensive benchmarks comparing all engine variants to demonstrate
//! the effectiveness of our mate-search-first + neural network approach.

use kingfisher::benchmarks::strength_testing::{StrengthTestConfig, StrengthTester};
use std::env;

fn main() {
    println!("🏆 Kingfisher Chess Engine - Comprehensive Strength Testing");
    println!("===========================================================");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args);

    println!("⚙️  Test Configuration:");
    println!("   Time per position: {}ms", config.time_limit_ms);
    println!("   MCTS iterations: {}", config.mcts_iterations);
    println!("   Alpha-beta depth: {}", config.ab_depth);
    println!("   Mate search depth: {}", config.mate_search_depth);
    if let Some(ref model_path) = config.neural_model_path {
        println!("   Neural model: {}", model_path);
    } else {
        println!("   Neural model: None (will skip neural variants)");
    }

    // Create and run strength tester
    let mut tester = StrengthTester::new(config);
    let results = tester.run_comprehensive_test();

    // Save detailed results
    if let Err(e) = results.save_to_csv("strength_test_results.csv") {
        println!("⚠️  Failed to save detailed results: {}", e);
    }

    println!("\n🎉 Strength testing complete!");
    println!("📁 Detailed results saved to: strength_test_results.csv");
    println!("\nKey takeaways:");
    println!("- Compare the rankings to see which approach works best");
    println!("- Look for positive improvement percentages in mate-search and neural network");
    println!("- Use these results to tune parameters and validate our innovations");
}

fn parse_args(args: &[String]) -> StrengthTestConfig {
    let mut config = StrengthTestConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--time" => {
                if i + 1 < args.len() {
                    if let Ok(time) = args[i + 1].parse::<u64>() {
                        config.time_limit_ms = time;
                    }
                    i += 1;
                }
            }
            "--iterations" => {
                if i + 1 < args.len() {
                    if let Ok(iters) = args[i + 1].parse::<u32>() {
                        config.mcts_iterations = iters;
                    }
                    i += 1;
                }
            }
            "--depth" => {
                if i + 1 < args.len() {
                    if let Ok(depth) = args[i + 1].parse::<i32>() {
                        config.ab_depth = depth;
                    }
                    i += 1;
                }
            }
            "--mate-depth" => {
                if i + 1 < args.len() {
                    if let Ok(depth) = args[i + 1].parse::<i32>() {
                        config.mate_search_depth = depth;
                    }
                    i += 1;
                }
            }
            "--neural-model" => {
                if i + 1 < args.len() {
                    config.neural_model_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--no-neural" => {
                config.neural_model_path = None;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                println!("⚠️  Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    config
}

fn print_help() {
    println!("Kingfisher Strength Testing");
    println!();
    println!("USAGE:");
    println!("    cargo run --bin strength_test [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --time <ms>           Time limit per position in milliseconds [default: 1000]");
    println!("    --iterations <n>      MCTS iterations limit [default: 500]");
    println!("    --depth <n>           Alpha-beta search depth [default: 6]");
    println!("    --mate-depth <n>      Mate search depth for MCTS [default: 3]");
    println!("    --neural-model <path> Path to neural network model [default: python/models/chess_model.pth]");
    println!("    --no-neural           Skip neural network variants");
    println!("    --help, -h            Show this help message");
    println!();
    println!("EXAMPLES:");
    println!("    # Quick test with reduced time");
    println!("    cargo run --bin strength_test --time 500");
    println!();
    println!("    # Thorough test with more iterations");
    println!("    cargo run --bin strength_test --time 2000 --iterations 1000");
    println!();
    println!("    # Test without neural network");
    println!("    cargo run --bin strength_test --no-neural");
    println!();
    println!("This benchmark tests 5 engine variants:");
    println!("  1. Alpha-Beta (baseline classical search)");
    println!("  2. MCTS-Classical (standard MCTS)");
    println!("  3. MCTS-Mate-Priority (our mate-search-first innovation)");
    println!("  4. MCTS-Neural (MCTS with neural network policy)");
    println!("  5. MCTS-Complete (mate-search-first + neural network)");
}
