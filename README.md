# Chapter 209: Data Distillation Trading

## 1. Introduction

Dataset distillation is a rapidly evolving area of machine learning that addresses a fundamental question: can we create a small synthetic dataset that, when used to train a model, produces performance comparable to training on the full dataset? Unlike traditional data compression or sampling techniques, dataset distillation synthesizes entirely new data points that encode the essential information of the original dataset in a highly condensed form.

The concept was first introduced by Wang et al. (2018) under the name "Dataset Distillation," drawing an analogy to knowledge distillation where a smaller student model learns to mimic a larger teacher model. In dataset distillation, instead of compressing the model, we compress the data itself. The distilled dataset consists of synthetic examples — often far fewer than the original — that are optimized so that a model trained on them generalizes as well as one trained on the full dataset.

For algorithmic trading, dataset distillation offers transformative potential. Financial markets generate enormous volumes of tick data, order book snapshots, and OHLCV candles across thousands of instruments. Training models on years of historical data is computationally expensive and time-consuming. If we could distill years of market history into a small set of representative synthetic samples — capturing bull runs, bear crashes, sideways consolidation, volatility spikes, and regime transitions — we could dramatically accelerate model development, backtesting, and retraining cycles.

This chapter explores the mathematical foundations of dataset distillation, surveys the key algorithms, and demonstrates practical applications for trading systems. We implement the core techniques in Rust, optimized for performance, and integrate real market data from the Bybit exchange.

## 2. Mathematical Foundation

Dataset distillation can be framed as a bilevel optimization problem. Given a large training set $\mathcal{T} = \{(x_i, y_i)\}_{i=1}^{N}$ and a model parameterized by $\theta$, we want to find a small synthetic dataset $\mathcal{S} = \{(\tilde{x}_j, \tilde{y}_j)\}_{j=1}^{M}$ where $M \ll N$, such that a model trained on $\mathcal{S}$ performs well on unseen data.

### 2.1 Gradient Matching

Gradient matching, introduced by Zhao et al. (2021) in Dataset Condensation (DC), is one of the most elegant approaches. The core idea is that if the gradients produced by the synthetic data match the gradients produced by the real data, then training on the synthetic data will follow a similar optimization trajectory.

Formally, we minimize the distance between gradients computed on real and synthetic data:

$$\min_{\mathcal{S}} \mathbb{E}_{\theta_0 \sim P_{\theta_0}} \sum_{t=0}^{T-1} D\left(\nabla_\theta \mathcal{L}(\mathcal{S}; \theta_t), \nabla_\theta \mathcal{L}(\mathcal{T}; \theta_t)\right)$$

where $D(\cdot, \cdot)$ is a distance function (typically cosine distance), $\theta_t$ represents model parameters at training step $t$, and $P_{\theta_0}$ is a distribution over random initializations.

The cosine distance is preferred over Euclidean distance because it is invariant to gradient magnitude, focusing instead on the direction of the update:

$$D_{\cos}(g_1, g_2) = 1 - \frac{g_1 \cdot g_2}{\|g_1\| \|g_2\|}$$

### 2.2 Distribution Matching

Distribution matching (Zhao & Bilen, 2023) takes a kernel-based approach. Instead of matching gradients, it matches the feature distributions of real and synthetic data in a learned embedding space:

$$\min_{\mathcal{S}} \left\| \frac{1}{|\mathcal{T}|} \sum_{x \in \mathcal{T}} \phi(x) - \frac{1}{|\mathcal{S}|} \sum_{\tilde{x} \in \mathcal{S}} \phi(\tilde{x}) \right\|^2$$

where $\phi(\cdot)$ is a feature extractor (often the penultimate layer of a neural network). This approach is computationally cheaper than gradient matching because it avoids second-order derivatives, though it may sacrifice some quality.

In the trading context, this means ensuring that the statistical properties of the distilled market data — return distributions, volatility clustering, correlation structures — match those of the full historical dataset.

### 2.3 Trajectory Matching

Trajectory matching, proposed by Cazenavette et al. (2022) in MTT (Matching Training Trajectories), optimizes synthetic data so that models trained on it follow the same parameter-space trajectory as models trained on real data:

$$\min_{\mathcal{S}} \mathbb{E} \left\| \theta_{t+N}^{\mathcal{S}} - \theta_{t+M}^{\mathcal{T}} \right\|^2$$

where $\theta_{t+N}^{\mathcal{S}}$ denotes parameters after $N$ steps on synthetic data starting from $\theta_t$, and $\theta_{t+M}^{\mathcal{T}}$ denotes parameters after $M$ steps on real data. This method captures longer-range training dynamics and often produces superior results.

### 2.4 Kernel Inducing Points

Kernel Inducing Points (KIP), introduced by Nguyen et al. (2021), leverages kernel ridge regression. Under the Neural Tangent Kernel (NTK) framework, the optimal synthetic dataset can be found by solving:

$$\min_{\mathcal{S}} \| K_{\mathcal{T}\mathcal{S}} (K_{\mathcal{S}\mathcal{S}} + \lambda I)^{-1} \tilde{Y} - Y_{\mathcal{T}} \|^2$$

where $K$ denotes kernel matrices. This has the advantage of having a closed-form inner-loop solution, making optimization more stable.

## 3. Dataset Distillation Algorithms

### 3.1 DD (Wang et al., 2018)

The original Dataset Distillation paper formulated the problem as learning synthetic data through backpropagation. The key insight was treating data points as learnable parameters. The algorithm:

1. Initialize synthetic images randomly
2. For each training step: (a) Initialize a fresh network, (b) Train the network on synthetic data for a few steps, (c) Evaluate the trained network on real data, (d) Backpropagate through the entire training process to update synthetic data

This requires unrolling the computational graph through the inner optimization loop, which is memory-intensive but produces high-quality distilled data.

### 3.2 Dataset Condensation (DC)

DC by Zhao et al. (2021) replaced the expensive bilevel optimization of DD with single-step gradient matching. This dramatically reduced computational cost while maintaining competitive performance. The algorithm samples random network initializations, computes gradients on both real and synthetic data, and updates the synthetic data to minimize gradient distance. DC is the approach we implement in this chapter due to its balance of simplicity and effectiveness.

### 3.3 CAFE (Aligning Features)

CAFE (Wang et al., 2022) combines feature alignment at multiple layers of a neural network. Rather than matching only final-layer features or gradients, CAFE aligns intermediate representations, capturing multi-scale information. For trading data, this means capturing both local patterns (individual candle shapes) and global patterns (trend structures, regime characteristics).

### 3.4 MTT (Matching Training Trajectories)

MTT by Cazenavette et al. (2022) pre-computes expert training trajectories on the full dataset and then optimizes synthetic data to reproduce these trajectories. It achieves state-of-the-art results on benchmarks but requires storing many expert trajectories, increasing storage requirements. For trading, expert trajectories could represent the learning process of a profitable strategy across different market conditions.

## 4. Trading Applications

### 4.1 Condensing Market History

Consider a trading system trained on 10 years of minute-bar data for 500 stocks — billions of data points. Dataset distillation can compress this into a synthetic dataset that captures the essential market dynamics: trending periods, mean-reverting periods, crash dynamics, recovery patterns, and liquidity regimes. A model trained on this condensed dataset would learn the same patterns as one trained on the full history, but in a fraction of the time.

### 4.2 Efficient Backtesting

Backtesting strategies across decades of data is slow. With distilled data representing the key market regimes, initial strategy evaluation can happen orders of magnitude faster. Only promising strategies need full-dataset backtesting, dramatically reducing the computational cost of strategy development.

### 4.3 Fast Model Retraining

In production trading, models often need frequent retraining as market conditions evolve. Maintaining a distilled dataset that is periodically updated allows rapid retraining — critical when deploying to live markets where speed matters. Instead of retraining on months of data, a model can be retrained on a distilled set in seconds.

### 4.4 Regime-Representative Data

Perhaps the most valuable application is creating synthetic data points that represent specific market regimes. By distilling data from identified regime clusters separately, we can create a balanced dataset that gives equal weight to rare but important regimes (like flash crashes) that would otherwise be underrepresented in standard training.

## 5. Practical Considerations

### 5.1 Distilled Data Interpretability

Unlike random subsets of real data, distilled synthetic data points are optimized artifacts. Individual distilled points may not look like any real market observation — they are superpositions of patterns that maximize information density. This can make them difficult to interpret visually but does not diminish their training utility.

In practice, we find that distilled market data points often exhibit exaggerated features: extreme volume spikes paired with specific price patterns, or unusual combinations of technical indicator values. These represent the model's learned "essence" of what matters in the data.

### 5.2 Storage Efficiency

A typical trading dataset with 5 years of minute-bar data for a single instrument contains approximately 1.3 million candles. If each candle has 6 features (OHLCV + timestamp), that is roughly 60 MB of float64 data. Distilling this to 20 synthetic points reduces storage to under 1 KB — a compression ratio exceeding 60,000x. This enables edge deployment, mobile trading applications, and efficient model versioning.

### 5.3 Update Strategies

Distilled datasets must be updated as market conditions evolve. Several strategies exist:

- **Periodic re-distillation**: Re-run the full distillation process weekly or monthly on a rolling window of data
- **Incremental updates**: Fine-tune existing distilled data by incorporating new market observations
- **Regime-triggered updates**: Re-distill only when a regime change is detected, preserving stable distilled data during steady market conditions
- **Ensemble distillation**: Maintain separate distilled datasets for different time horizons (intraday, daily, weekly) and combine them for training

## 6. Implementation Walkthrough

Our Rust implementation provides a complete dataset distillation pipeline for trading data. The core components are:

### Data Representation

We represent market data as feature vectors. Each candle is transformed into a normalized feature vector containing returns, volatility measures, volume ratios, and price range metrics. The `MarketData` struct holds raw OHLCV data, while `DistillationConfig` controls the distillation process.

### Gradient Matching Engine

The heart of the implementation is the gradient matching loop. We maintain a set of learnable synthetic data points (initialized randomly or from cluster centroids) and a simple linear model. At each iteration:

1. Compute the gradient of the loss on a batch of real data
2. Compute the gradient of the loss on the synthetic data
3. Compute cosine distance between the gradient vectors
4. Update synthetic data points to minimize this distance

### Model Training and Evaluation

We implement a simple linear regression model suitable for predicting next-candle returns. The model is deliberately simple — dataset distillation is most powerful when the model class is fixed and the goal is to find the most informative training data for that model class. Our evaluation compares three approaches: training on full data, training on distilled data, and training on a randomly sampled subset of the same size as the distilled data.

### Coreset Selection Baseline

As a comparison, we implement k-medoids coreset selection, which picks real data points that best represent the dataset geometry. This provides a strong baseline — if distillation does not outperform simple coreset selection, the additional complexity is not justified.

```rust
// Core distillation loop (simplified)
for epoch in 0..config.epochs {
    let real_grad = compute_gradient(&model, &real_batch);
    let synth_grad = compute_gradient(&model, &synthetic_data);
    let cos_dist = cosine_distance(&real_grad, &synth_grad);
    update_synthetic_data(&mut synthetic_data, cos_dist, config.lr);
}
```

The full implementation handles mini-batching, learning rate scheduling, multiple random initializations, and convergence checking. See `rust/src/lib.rs` for the complete code.

## 7. Bybit Data Integration

We integrate with the Bybit exchange API to fetch real-time and historical OHLCV data. The implementation uses Bybit's v5 REST API:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=15&limit=200
```

The `BybitClient` struct in our implementation handles:

- Fetching historical kline (candlestick) data
- Parsing JSON responses into `MarketData` structs
- Rate limiting and error handling
- Support for multiple intervals (1m, 5m, 15m, 1h, 4h, 1d)

Data from Bybit is preprocessed before distillation: prices are log-transformed, volumes are normalized, and features are standardized to zero mean and unit variance. This normalization is critical for gradient matching to work correctly, as features on vastly different scales would dominate the gradient computation.

```rust
let client = BybitClient::new();
let candles = client.fetch_klines("BTCUSDT", "15", 200).await?;
let features = preprocess_candles(&candles);
let distilled = distill_dataset(&features, &config)?;
```

## 8. Key Takeaways

1. **Dataset distillation creates synthetic training data** that encodes the essential information of a full dataset in far fewer examples. This is fundamentally different from sampling or compression.

2. **Gradient matching is the most practical approach** for trading applications, offering a good balance between computational cost and distillation quality. It ensures models trained on synthetic data follow similar optimization paths.

3. **Trading applications are compelling**: condensing years of market data into tens of synthetic points enables rapid backtesting, fast retraining, and efficient edge deployment.

4. **Distilled data captures market regimes**: synthetic points naturally encode regime information, with each point representing a blend of market conditions that maximizes training utility.

5. **Update strategies matter**: distilled datasets must evolve with markets. Regime-triggered re-distillation offers the best balance of freshness and stability.

6. **Quality metrics are essential**: always validate distilled data by comparing train-on-distilled/test-on-real performance against full-data and random-subset baselines.

7. **Compression ratios are extreme**: reducing millions of candles to tens of synthetic points (60,000x+ compression) enables novel deployment scenarios impossible with full datasets.

8. **Rust implementation provides production-grade performance**: the computational cost of distillation itself is amortized over many subsequent fast training runs, making the upfront investment worthwhile.

Dataset distillation represents a paradigm shift in how we think about training data for trading models. Rather than asking "how much data do we need?", we ask "what is the minimum synthetic dataset that captures all the information our model can use?" This shift enables faster iteration, more efficient resource usage, and ultimately better trading systems.

## References

- Wang, T., Zhu, J.Y., Torralba, A., & Efros, A.A. (2018). Dataset Distillation. arXiv:1811.10959.
- Zhao, B., Mopuri, K.R., & Bilen, H. (2021). Dataset Condensation with Gradient Matching. ICLR 2021.
- Zhao, B. & Bilen, H. (2023). Dataset Condensation with Distribution Matching. WACV 2023.
- Cazenavette, G., Wang, T., Torralba, A., Efros, A.A., & Zhu, J.Y. (2022). Dataset Distillation by Matching Training Trajectories. CVPR 2022.
- Nguyen, T., Chen, Z., & Lee, J. (2021). Dataset Meta-Learning from Kernel Ridge-Regression. ICLR 2021.
- Wang, K., Zhao, B., Peng, X., Zhu, Z., Yang, S., Wang, S., Huang, G., Bilen, H., Wang, X., & You, Y. (2022). CAFE: Learning to Condense Dataset by Aligning Features. CVPR 2022.
