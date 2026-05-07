**Idea: Rank-Based Entropy Exploration**

Uses rank-transformed success rates (robust to outliers) combined with entropy-based exploration and diversity-sensitive perturbation. This addresses near-optimum tasks (like Task 12 at 7.3e-05) where greedy success tracking is too exploitative and destabilizes nearly-converged solutions.

```python
def _adapt_strategy_probabilities(self):
        # Compute rank-based probabilities (robust to outliers, helps near-optimum tasks)
        if np.any(self.strategy_success_count > 0):
            success_rate = self.strategy_success_count / (self.strategy_attempt_count + 1e-10)
            ranks = np.argsort(np.argsort(-success_rate)) + 1
            rank_probs = ranks / (np.sum(ranks) + 1e-10)
        else:
            rank_probs = np.ones(self.n_strategies) / self.n_strategies

        # Entropy-based exploration temperature (adapts to stagnation)
        stagnation_factor = min(1.0, self.generation_without_improvement / 20.0)
        temperature = 0.5 + 1.5 * stagnation_factor
        entropy_weight = 0.2 + 0.4 * stagnation_factor

        # Diversity-sensitive perturbation (higher diversity = more exploration)
        if len(self.diversity_history) >= 10:
            recent_div = np.mean(self.diversity_history[-10:])
            div_range = max(np.max(self.diversity_history[-50:]) - np.min(self.diversity_history[-50:]), 1e-10)
            diversity_signal = np.clip((recent_div - np.min(self.diversity_history[-50:])) / div_range, 0.0, 1.0)
        else:
            diversity_signal = 0.5

        entropy_weight = np.clip(entropy_weight * (1.0 - 0.5 * diversity_signal), 0.1, 0.6)

        # Softmax with temperature for controlled randomness
        logits = np.log(rank_probs + 1e-10) / temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        exploration_probs = exp_logits / (np.sum(exp_logits) + 1e-10)

        # Combine exploitation (rank-based) with exploration (entropy-driven)
        final_probs = (1.0 - entropy_weight) * rank_probs + entropy_weight * exploration_probs

        # Numerical safety
        final_probs = np.clip(final_probs, 1e-10, 1.0)
        final_probs = final_probs / (np.sum(final_probs) + 1e-10)

        # Assign strategies with minimal disruption near optimum
        for i in range(self.NP):
            self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=final_probs)
```