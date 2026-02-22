//! # Data Distillation Trading
//!
//! Dataset distillation for trading: creating small synthetic datasets
//! that train models as effectively as full market datasets.
//!
//! Implements gradient matching based dataset distillation,
//! coreset selection, and Bybit API integration.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Raw OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Configuration for the distillation process.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Number of synthetic data points to produce.
    pub num_distilled: usize,
    /// Number of features per data point.
    pub num_features: usize,
    /// Learning rate for synthetic data optimisation.
    pub lr: f64,
    /// Number of outer-loop epochs.
    pub epochs: usize,
    /// Mini-batch size when sampling real data.
    pub batch_size: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            num_distilled: 15,
            num_features: 5,
            lr: 0.01,
            epochs: 200,
            batch_size: 32,
            seed: 42,
        }
    }
}

/// Result of a distillation run.
#[derive(Debug, Clone)]
pub struct DistillationResult {
    /// Distilled synthetic feature matrix (num_distilled x num_features).
    pub synthetic_features: Array2<f64>,
    /// Distilled synthetic labels.
    pub synthetic_labels: Array1<f64>,
    /// Final gradient matching loss.
    pub final_loss: f64,
}

/// Evaluation metrics comparing different training approaches.
#[derive(Debug, Clone)]
pub struct EvalMetrics {
    /// MSE when training on the full dataset.
    pub full_data_mse: f64,
    /// MSE when training on distilled data.
    pub distilled_data_mse: f64,
    /// MSE when training on a random subset of the same size.
    pub random_subset_mse: f64,
    /// MSE when training on a coreset of the same size.
    pub coreset_mse: f64,
}

impl std::fmt::Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Evaluation Metrics (MSE, lower is better) ===")?;
        writeln!(f, "  Full data:      {:.6}", self.full_data_mse)?;
        writeln!(f, "  Distilled data: {:.6}", self.distilled_data_mse)?;
        writeln!(f, "  Random subset:  {:.6}", self.random_subset_mse)?;
        writeln!(f, "  Coreset:        {:.6}", self.coreset_mse)
    }
}

// ---------------------------------------------------------------------------
// Feature engineering
// ---------------------------------------------------------------------------

/// Convert raw candles into a feature matrix and label vector.
///
/// Features per row (computed from consecutive candles):
///   0: log-return  = ln(close_t / close_{t-1})
///   1: high-low range normalised by close
///   2: body ratio  = (close - open) / (high - low + eps)
///   3: upper shadow ratio
///   4: log-volume ratio = ln(volume_t / volume_{t-1} + eps)
///
/// Label: next-candle log-return.
pub fn candles_to_features(candles: &[Candle]) -> Result<(Array2<f64>, Array1<f64>)> {
    if candles.len() < 3 {
        return Err(anyhow!("Need at least 3 candles to build features"));
    }

    let n = candles.len() - 2; // we lose one at each end
    let nf = 5usize;
    let mut features = Array2::<f64>::zeros((n, nf));
    let mut labels = Array1::<f64>::zeros(n);

    for i in 0..n {
        let prev = &candles[i];
        let curr = &candles[i + 1];
        let next = &candles[i + 2];

        let eps = 1e-10;

        // log-return
        let log_ret = (curr.close / (prev.close + eps)).ln();
        // range
        let range = (curr.high - curr.low) / (curr.close + eps);
        // body ratio
        let hl = curr.high - curr.low + eps;
        let body = (curr.close - curr.open) / hl;
        // upper shadow
        let upper_shadow = (curr.high - curr.close.max(curr.open)) / hl;
        // volume ratio
        let vol_ratio = ((curr.volume + eps) / (prev.volume + eps)).ln();

        features[[i, 0]] = log_ret;
        features[[i, 1]] = range;
        features[[i, 2]] = body;
        features[[i, 3]] = upper_shadow;
        features[[i, 4]] = vol_ratio;

        // label: next log-return
        labels[i] = (next.close / (curr.close + eps)).ln();
    }

    Ok((features, labels))
}

/// Standardise each column to zero mean and unit variance. Returns (normalised, means, stds).
pub fn standardise(data: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let means = data.mean_axis(Axis(0)).unwrap();
    let stds = data.std_axis(Axis(0), 0.0);

    let ncols = data.ncols();
    let mut normed = data.clone();
    for c in 0..ncols {
        let mean = means[c];
        let s = if stds[c] < 1e-12 { 1.0 } else { stds[c] };
        for r in 0..normed.nrows() {
            normed[[r, c]] = (normed[[r, c]] - mean) / s;
        }
    }

    (normed, means, stds)
}

// ---------------------------------------------------------------------------
// Simple linear model
// ---------------------------------------------------------------------------

/// A minimal linear regression model: y = X * w + b.
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Array1<f64>,
    pub bias: f64,
}

impl LinearModel {
    /// Random initialisation.
    pub fn random(num_features: usize, rng: &mut impl Rng) -> Self {
        let weights = Array1::from_iter((0..num_features).map(|_| rng.gen_range(-0.1..0.1)));
        let bias = rng.gen_range(-0.01..0.01);
        Self { weights, bias }
    }

    /// Predict.
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.weights) + self.bias
    }

    /// Mean squared error.
    pub fn mse(&self, x: &Array2<f64>, y: &Array1<f64>) -> f64 {
        let pred = self.predict(x);
        let diff = &pred - y;
        diff.mapv(|v| v * v).mean().unwrap_or(f64::MAX)
    }

    /// Compute gradient of MSE w.r.t. (weights, bias).
    /// Returns concatenated vector [dw_0, ..., dw_{d-1}, db].
    pub fn gradient(&self, x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
        let n = x.nrows() as f64;
        let pred = self.predict(x);
        let residual = &pred - y; // (n,)

        // dL/dw = (2/n) * X^T * residual
        let dw = x.t().dot(&residual) * (2.0 / n);

        // dL/db = (2/n) * sum(residual)
        let db = residual.sum() * (2.0 / n);

        let mut grad = Array1::zeros(self.weights.len() + 1);
        grad.slice_mut(ndarray::s![..self.weights.len()])
            .assign(&dw);
        grad[self.weights.len()] = db;
        grad
    }

    /// Single gradient-descent step.
    pub fn step(&mut self, x: &Array2<f64>, y: &Array1<f64>, lr: f64) {
        let grad = self.gradient(x, y);
        let d = self.weights.len();
        for i in 0..d {
            self.weights[i] -= lr * grad[i];
        }
        self.bias -= lr * grad[d];
    }

    /// Train for a number of epochs with given learning rate.
    pub fn train(&mut self, x: &Array2<f64>, y: &Array1<f64>, epochs: usize, lr: f64) {
        for _ in 0..epochs {
            self.step(x, y, lr);
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient matching distillation
// ---------------------------------------------------------------------------

/// Cosine distance between two 1-D arrays.
pub fn cosine_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.mapv(|v| v * v).sum().sqrt();
    let nb = b.mapv(|v| v * v).sum().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

/// Distribution matching loss: squared difference of feature-wise means.
pub fn distribution_matching_loss(real: &Array2<f64>, synthetic: &Array2<f64>) -> f64 {
    let real_mean = real.mean_axis(Axis(0)).unwrap();
    let synth_mean = synthetic.mean_axis(Axis(0)).unwrap();
    let diff = &real_mean - &synth_mean;
    diff.mapv(|v| v * v).sum()
}

/// Run gradient-matching dataset distillation.
///
/// Produces a small synthetic dataset that induces similar gradients on a
/// linear model as the real dataset does.
pub fn distill_dataset(
    features: &Array2<f64>,
    labels: &Array1<f64>,
    config: &DistillationConfig,
) -> Result<DistillationResult> {
    if features.nrows() < config.batch_size {
        return Err(anyhow!(
            "Dataset has {} rows but batch_size is {}",
            features.nrows(),
            config.batch_size
        ));
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let nf = features.ncols();

    // Initialise synthetic data from random real samples
    let mut synth_x = Array2::<f64>::zeros((config.num_distilled, nf));
    let mut synth_y = Array1::<f64>::zeros(config.num_distilled);
    for i in 0..config.num_distilled {
        let idx = rng.gen_range(0..features.nrows());
        synth_x.row_mut(i).assign(&features.row(idx));
        synth_y[i] = labels[idx];
    }

    let mut final_loss = f64::MAX;

    for _epoch in 0..config.epochs {
        // Fresh random model each outer step (following DC)
        let model = LinearModel::random(nf, &mut rng);

        // Sample a mini-batch of real data
        let batch_indices: Vec<usize> = (0..config.batch_size)
            .map(|_| rng.gen_range(0..features.nrows()))
            .collect();

        let batch_x = features.select(Axis(0), &batch_indices);
        let batch_y = labels.select(Axis(0), &batch_indices);

        // Gradients on real and synthetic data
        let real_grad = model.gradient(&batch_x, &batch_y);
        let synth_grad = model.gradient(&synth_x, &synth_y);

        // Cosine distance
        let cos_dist = cosine_distance(&real_grad, &synth_grad);
        final_loss = cos_dist;

        // Compute numerical gradient of cosine distance w.r.t. each synthetic point
        let perturbation = 1e-5;
        for i in 0..config.num_distilled {
            for j in 0..nf {
                let orig = synth_x[[i, j]];

                synth_x[[i, j]] = orig + perturbation;
                let grad_plus = model.gradient(&synth_x, &synth_y);
                let loss_plus = cosine_distance(&real_grad, &grad_plus);

                synth_x[[i, j]] = orig - perturbation;
                let grad_minus = model.gradient(&synth_x, &synth_y);
                let loss_minus = cosine_distance(&real_grad, &grad_minus);

                synth_x[[i, j]] = orig;

                let dloss = (loss_plus - loss_minus) / (2.0 * perturbation);
                synth_x[[i, j]] -= config.lr * dloss;
            }

            // Also update synthetic label
            let orig_y = synth_y[i];

            synth_y[i] = orig_y + perturbation;
            let grad_plus = model.gradient(&synth_x, &synth_y);
            let loss_plus = cosine_distance(&real_grad, &grad_plus);

            synth_y[i] = orig_y - perturbation;
            let grad_minus = model.gradient(&synth_x, &synth_y);
            let loss_minus = cosine_distance(&real_grad, &grad_minus);

            synth_y[i] = orig_y;

            let dloss = (loss_plus - loss_minus) / (2.0 * perturbation);
            synth_y[i] -= config.lr * dloss;
        }
    }

    Ok(DistillationResult {
        synthetic_features: synth_x,
        synthetic_labels: synth_y,
        final_loss,
    })
}

// ---------------------------------------------------------------------------
// Coreset selection (k-medoids style)
// ---------------------------------------------------------------------------

/// Select a coreset of `k` points from the dataset using a greedy k-medoids approach.
///
/// Returns indices of selected points.
pub fn select_coreset(features: &Array2<f64>, k: usize, seed: u64) -> Vec<usize> {
    let n = features.nrows();
    if k >= n {
        return (0..n).collect();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut medoids: Vec<usize> = Vec::with_capacity(k);

    // Start with a random point
    medoids.push(rng.gen_range(0..n));

    // Greedily add the point that is farthest from current medoids (k-center)
    for _ in 1..k {
        let mut best_idx = 0;
        let mut best_dist = f64::NEG_INFINITY;

        for i in 0..n {
            if medoids.contains(&i) {
                continue;
            }
            let row_i = features.row(i);
            let min_dist = medoids
                .iter()
                .map(|&m| {
                    let diff = &row_i - &features.row(m);
                    diff.mapv(|v| v * v).sum().sqrt()
                })
                .fold(f64::INFINITY, f64::min);
            if min_dist > best_dist {
                best_dist = min_dist;
                best_idx = i;
            }
        }
        medoids.push(best_idx);
    }

    medoids
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Full evaluation: train-on-X, test-on-held-out for each approach.
pub fn evaluate(
    train_x: &Array2<f64>,
    train_y: &Array1<f64>,
    test_x: &Array2<f64>,
    test_y: &Array1<f64>,
    distill_result: &DistillationResult,
    config: &DistillationConfig,
) -> EvalMetrics {
    let mut rng = StdRng::seed_from_u64(config.seed + 100);
    let nf = train_x.ncols();
    let train_epochs = 200;
    let model_lr = 0.01;

    // 1. Full data
    let mut model_full = LinearModel::random(nf, &mut rng);
    model_full.train(train_x, train_y, train_epochs, model_lr);
    let full_data_mse = model_full.mse(test_x, test_y);

    // 2. Distilled data
    let mut model_distilled = LinearModel::random(nf, &mut rng);
    model_distilled.train(
        &distill_result.synthetic_features,
        &distill_result.synthetic_labels,
        train_epochs,
        model_lr,
    );
    let distilled_data_mse = model_distilled.mse(test_x, test_y);

    // 3. Random subset
    let subset_size = config.num_distilled.min(train_x.nrows());
    let subset_indices: Vec<usize> = (0..subset_size)
        .map(|_| rng.gen_range(0..train_x.nrows()))
        .collect();
    let subset_x = train_x.select(Axis(0), &subset_indices);
    let subset_y = train_y.select(Axis(0), &subset_indices);

    let mut model_random = LinearModel::random(nf, &mut rng);
    model_random.train(&subset_x, &subset_y, train_epochs, model_lr);
    let random_subset_mse = model_random.mse(test_x, test_y);

    // 4. Coreset
    let coreset_indices = select_coreset(train_x, config.num_distilled, config.seed + 200);
    let coreset_x = train_x.select(Axis(0), &coreset_indices);
    let coreset_y = train_y.select(Axis(0), &coreset_indices);

    let mut model_coreset = LinearModel::random(nf, &mut rng);
    model_coreset.train(&coreset_x, &coreset_y, train_epochs, model_lr);
    let coreset_mse = model_coreset.mse(test_x, test_y);

    EvalMetrics {
        full_data_mse,
        distilled_data_mse,
        random_subset_mse,
        coreset_mse,
    }
}

// ---------------------------------------------------------------------------
// Bybit API client
// ---------------------------------------------------------------------------

/// Bybit REST API v5 response types.
#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub list: Vec<Vec<String>>,
}

/// Client for fetching OHLCV data from Bybit.
pub struct BybitClient {
    base_url: String,
}

impl BybitClient {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Fetch kline (candlestick) data.
    ///
    /// * `symbol` – e.g. "BTCUSDT"
    /// * `interval` – "1", "5", "15", "60", "240", "D"
    /// * `limit` – max number of candles (up to 200)
    pub async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::get(&url).await?.json().await?;

        if resp.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        // Bybit returns newest first; reverse to chronological order
        candles.reverse();
        Ok(candles)
    }

    /// Blocking version of `fetch_klines` for non-async contexts.
    pub fn fetch_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Candle>> {
        let url = format!(
            "{}/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let resp: BybitResponse = reqwest::blocking::get(&url)?.json()?;

        if resp.ret_code != 0 {
            return Err(anyhow!("Bybit API error: {}", resp.ret_msg));
        }

        let mut candles: Vec<Candle> = resp
            .result
            .list
            .iter()
            .filter_map(|row| {
                if row.len() < 6 {
                    return None;
                }
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            })
            .collect();

        candles.reverse();
        Ok(candles)
    }
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate synthetic candle data for testing without network access.
pub fn generate_synthetic_candles(n: usize, seed: u64) -> Vec<Candle> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0_f64; // starting price

    for i in 0..n {
        let ret = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = open * (1.0 + ret);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1_700_000_000 + (i as u64) * 900,
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candles() -> Vec<Candle> {
        generate_synthetic_candles(100, 42)
    }

    #[test]
    fn test_candles_to_features() {
        let candles = make_test_candles();
        let (features, labels) = candles_to_features(&candles).unwrap();
        assert_eq!(features.nrows(), candles.len() - 2);
        assert_eq!(features.ncols(), 5);
        assert_eq!(labels.len(), candles.len() - 2);
    }

    #[test]
    fn test_standardise() {
        let candles = make_test_candles();
        let (features, _) = candles_to_features(&candles).unwrap();
        let (normed, _means, _stds) = standardise(&features);
        assert_eq!(normed.shape(), features.shape());

        // Each column should have mean ~0
        for col in normed.columns() {
            let mean = col.mean().unwrap();
            assert!(mean.abs() < 1e-10, "Column mean should be ~0, got {}", mean);
        }
    }

    #[test]
    fn test_linear_model_gradient() {
        let mut rng = StdRng::seed_from_u64(123);
        let model = LinearModel::random(3, &mut rng);
        let x = Array2::from_shape_vec((4, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let grad = model.gradient(&x, &y);
        assert_eq!(grad.len(), 4); // 3 weights + 1 bias
    }

    #[test]
    fn test_linear_model_train() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = LinearModel::random(1, &mut rng);

        // Simple data: y = 2x + 1
        let x = Array2::from_shape_vec((5, 1), vec![1., 2., 3., 4., 5.]).unwrap();
        let y = Array1::from_vec(vec![3., 5., 7., 9., 11.]);

        let mse_before = model.mse(&x, &y);
        model.train(&x, &y, 500, 0.01);
        let mse_after = model.mse(&x, &y);

        assert!(mse_after < mse_before, "Training should reduce MSE");
        assert!(mse_after < 0.1, "MSE should be small after training");
    }

    #[test]
    fn test_cosine_distance() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        let dist = cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-10, "Orthogonal vectors: dist=1");

        let c = Array1::from_vec(vec![1.0, 0.0]);
        let dist2 = cosine_distance(&a, &c);
        assert!(dist2.abs() < 1e-10, "Same direction: dist=0");
    }

    #[test]
    fn test_distribution_matching() {
        let a = Array2::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = a.clone();
        let loss = distribution_matching_loss(&a, &b);
        assert!(loss < 1e-10, "Identical distributions should have 0 loss");
    }

    #[test]
    fn test_distillation() {
        let candles = generate_synthetic_candles(120, 42);
        let (features, labels) = candles_to_features(&candles).unwrap();
        let (normed, _, _) = standardise(&features);

        let config = DistillationConfig {
            num_distilled: 10,
            num_features: 5,
            lr: 0.005,
            epochs: 50,
            batch_size: 16,
            seed: 42,
        };

        let result = distill_dataset(&normed, &labels, &config).unwrap();
        assert_eq!(result.synthetic_features.nrows(), 10);
        assert_eq!(result.synthetic_features.ncols(), 5);
        assert_eq!(result.synthetic_labels.len(), 10);
    }

    #[test]
    fn test_coreset_selection() {
        let candles = generate_synthetic_candles(50, 42);
        let (features, _) = candles_to_features(&candles).unwrap();

        let coreset = select_coreset(&features, 5, 42);
        assert_eq!(coreset.len(), 5);

        // All indices should be unique
        let mut sorted = coreset.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);
    }

    #[test]
    fn test_evaluation() {
        let candles = generate_synthetic_candles(150, 42);
        let (features, labels) = candles_to_features(&candles).unwrap();
        let (normed, _, _) = standardise(&features);

        let split = normed.nrows() * 3 / 4;
        let train_x = normed.slice(ndarray::s![..split, ..]).to_owned();
        let train_y = labels.slice(ndarray::s![..split]).to_owned();
        let test_x = normed.slice(ndarray::s![split.., ..]).to_owned();
        let test_y = labels.slice(ndarray::s![split..]).to_owned();

        let config = DistillationConfig {
            num_distilled: 10,
            num_features: 5,
            lr: 0.005,
            epochs: 30,
            batch_size: 16,
            seed: 42,
        };

        let result = distill_dataset(&train_x, &train_y, &config).unwrap();
        let metrics = evaluate(&train_x, &train_y, &test_x, &test_y, &result, &config);

        // All MSE values should be finite
        assert!(metrics.full_data_mse.is_finite());
        assert!(metrics.distilled_data_mse.is_finite());
        assert!(metrics.random_subset_mse.is_finite());
        assert!(metrics.coreset_mse.is_finite());
    }

    #[test]
    fn test_generate_synthetic_candles() {
        let candles = generate_synthetic_candles(50, 99);
        assert_eq!(candles.len(), 50);
        for c in &candles {
            assert!(c.high >= c.low);
            assert!(c.volume > 0.0);
        }
    }
}
