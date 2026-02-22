//! # Trading Example — Data Distillation
//!
//! Fetches BTCUSDT candles from Bybit, distills the dataset down to a small
//! set of synthetic data points, then compares prediction quality of models
//! trained on full data, distilled data, random subset, and coreset.
//!
//! Run:
//! ```sh
//! cargo run --example trading_example
//! ```

use anyhow::Result;
use data_distillation_trading::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Chapter 209: Data Distillation Trading ===\n");

    // ------------------------------------------------------------------
    // 1. Fetch data from Bybit (fall back to synthetic if network fails)
    // ------------------------------------------------------------------
    println!("[1] Fetching BTCUSDT 15m candles from Bybit...");

    let candles = match BybitClient::new()
        .fetch_klines("BTCUSDT", "15", 200)
        .await
    {
        Ok(c) if c.len() >= 50 => {
            println!("    Fetched {} candles from Bybit.\n", c.len());
            c
        }
        Ok(c) => {
            println!(
                "    Only got {} candles from Bybit, using synthetic data instead.\n",
                c.len()
            );
            generate_synthetic_candles(200, 42)
        }
        Err(e) => {
            println!("    Bybit API error: {}. Using synthetic data.\n", e);
            generate_synthetic_candles(200, 42)
        }
    };

    // ------------------------------------------------------------------
    // 2. Feature engineering
    // ------------------------------------------------------------------
    println!("[2] Building features...");
    let (features, labels) = candles_to_features(&candles)?;
    let (normed, _means, _stds) = standardise(&features);
    println!(
        "    {} samples, {} features each.\n",
        normed.nrows(),
        normed.ncols()
    );

    // Train/test split (75/25)
    let split = normed.nrows() * 3 / 4;
    let train_x = normed.slice(ndarray::s![..split, ..]).to_owned();
    let train_y = labels.slice(ndarray::s![..split]).to_owned();
    let test_x = normed.slice(ndarray::s![split.., ..]).to_owned();
    let test_y = labels.slice(ndarray::s![split..]).to_owned();

    println!(
        "    Train: {} samples, Test: {} samples\n",
        train_x.nrows(),
        test_x.nrows()
    );

    // ------------------------------------------------------------------
    // 3. Distillation
    // ------------------------------------------------------------------
    let num_distilled = 15;
    println!(
        "[3] Distilling {} training samples -> {} synthetic points...",
        train_x.nrows(),
        num_distilled
    );

    let config = DistillationConfig {
        num_distilled,
        num_features: 5,
        lr: 0.01,
        epochs: 200,
        batch_size: 32,
        seed: 42,
    };

    let result = distill_dataset(&train_x, &train_y, &config)?;
    println!("    Final gradient matching loss: {:.6}\n", result.final_loss);

    // ------------------------------------------------------------------
    // 4. Show distilled data
    // ------------------------------------------------------------------
    println!("[4] Distilled synthetic data points:");
    println!("    {:>5} {:>10} {:>10} {:>10} {:>10} {:>10}  {:>10}",
        "idx", "feat_0", "feat_1", "feat_2", "feat_3", "feat_4", "label");
    for i in 0..result.synthetic_features.nrows() {
        print!("    {:>5}", i);
        for j in 0..result.synthetic_features.ncols() {
            print!(" {:>10.4}", result.synthetic_features[[i, j]]);
        }
        println!("  {:>10.6}", result.synthetic_labels[i]);
    }
    println!();

    // ------------------------------------------------------------------
    // 5. Distribution matching check
    // ------------------------------------------------------------------
    let dm_loss = distribution_matching_loss(&train_x, &result.synthetic_features);
    println!(
        "[5] Distribution matching loss (real vs distilled): {:.6}\n",
        dm_loss
    );

    // ------------------------------------------------------------------
    // 6. Evaluation
    // ------------------------------------------------------------------
    println!("[6] Evaluating: train on each approach, test on held-out data...\n");

    let metrics = evaluate(&train_x, &train_y, &test_x, &test_y, &result, &config);
    println!("{}", metrics);

    // ------------------------------------------------------------------
    // 7. Compression stats
    // ------------------------------------------------------------------
    let full_size = train_x.nrows() * train_x.ncols() * 8; // f64 = 8 bytes
    let distilled_size = num_distilled * train_x.ncols() * 8;
    let ratio = full_size as f64 / distilled_size as f64;

    println!("[7] Compression statistics:");
    println!("    Full training data:  {} bytes ({} samples)", full_size, train_x.nrows());
    println!("    Distilled data:      {} bytes ({} samples)", distilled_size, num_distilled);
    println!("    Compression ratio:   {:.1}x\n", ratio);

    println!("=== Done ===");
    Ok(())
}
