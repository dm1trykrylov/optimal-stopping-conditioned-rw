# Optimal Stopping for a Conditioned Random Walk

This project studies analytical and machine learning approaches to the optimal stopping problem for a discrete random walk conditioned to stay between linear boundaries. The model is motivated by asset price dynamics constrained within a price corridor.

## Problem Statement

Given a symmetric discrete-time random walk constrained to remain within linear bounds, determine an optimal stopping rule (sell vs. hold) that maximizes the expected profit.

## Methods

### Analytical Approach
- Formalization of the optimal stopping problem for a conditioned random walk
- Derivation of the optimal stopping rule
- Recursive equations for the value function
- Theoretical analysis and proofs

### Machine Learning Approach
- GRU-based neural network
- Predicts future price dynamics
- Outputs sell confidence to maximize expected profit
- Trained on synthetic data generated for the conditioned process

## Key Results

Both approaches were evaluated on synthetic datasets consisting of discrete random walks conditioned to stay within linear boundaries. Performance was measured using the average achieved profit.

### Quantitative Comparison

| Trajectory Length (N) | Epochs | ML (GRU) Avg. Profit | Theoretical Avg. Profit |
|----------------------|--------|----------------------|-------------------------|
| 100                  | 50     | 0.79                 | 0.82                    |
| 200                  | 50     | 0.73                 | 0.76                    |
| 400                  | 50     | 0.55                 | 0.74                    |
| 400                  | 100    | 0.74                 | 0.74                    |

**Key observations:**
- The analytical optimal stopping rule provides a stable upper benchmark.
- The GRU-based model learns a competitive strategy without explicit knowledge of the optimal rule.
- Longer trajectories significantly increase learning difficulty.
- Additional training allows the ML approach to match theoretical performance.

### Profit vs. Decision Threshold

For different trajectory lengths \(N\), the average profit is evaluated as a function of the decision threshold:

- [N=100](images/profit_vs_threshold_validation_N100.png)
- [N=200](images/profit_vs_threshold_validation_N200.png)
- [N=400, epochs = 50](images/profit_vs_threshold_validation_N400.png)
- [N=400, epochs = 100](images/profit_vs_threshold_validation_N400b.png)

The plots illustrate robustness of the theoretical threshold and convergence of the learned strategy toward the analytical optimum.

## Project Artifacts

- Slides with project motivation and introduction (in Russian): [intro-slides](intro-slides.pdf)
- Final report with experiments and results (in Russian): [final-report](final-report.pdf), [final-slides](final-slides.pdf)
- Additional theoretical report outlining future research directions (in Russian): [report-slides](report-slides.pdf)

## Contributions

Dmitry Krylov (@dm1trykrylov):
- Analytical formulation and solution of the optimal stopping problem
- Derivation of recursive value functions
- Theoretical analysis and proofs

Andrew Strelkov (@asesen):
- Machine learning implementation and experiments

## Skills Demonstrated

- Stochastic processes
- Optimal stopping theory
- Dynamic programming
- Deep learning (GRU, PyTorch)
- Simulation and benchmarking
