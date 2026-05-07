**Idea: Multi-Dimensional Rank-Based Credit Assignment**

A fundamentally different credit assignment approach using rank-based rewards and multi-dimensional feedback (fitness improvement, population spread, best-fitness rank) instead of raw log-scaled improvement. This provides more stable gradient signals for the Thompson Sampling bandit, especially on deceptive landscapes where raw improvements are rare or noisy.

```python
def _update_operator_rewards(self):
        """Rank-based multi-dimensional credit assignment for stable operator selection."""
        # Initialize tracking structures
        if not hasattr(self, 'operator_rank_scores'):
            self.operator_rank_scores = {i: [] for i in range(self.num_operators)}
        if not hasattr(self, 'operator_spread_scores'):
            self.operator_spread_scores = {i: [] for i in range(self.num_operators)}
        if not hasattr(self, 'operator_improvement_scores'):
            self.operator_improvement_scores = {i: [] for i in range(self.num_operators)}

        # === DIMENSION 1: Fitness Improvement ===
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        if improvement > 0:
            # Log-scaled with temperature scaling based on problem difficulty
            temp = max(abs(self.f_opt), 1.0)
            imp_reward = float(np.log1p(improvement / (temp + 1e-10)) / np.log(11.0))
        else:
            imp_reward = 0.0

        # === DIMENSION 2: Population Spread (diversity contribution) ===
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        spread_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        spread_reward = float(spread_ratio)

        # === DIMENSION 3: Best-Fitness Rank in Population ===
        # Compare current best to population distribution
        fit_mean = np.mean(self.fitness)
        fit_std = max(np.std(self.fitness), 1e-10)
        if fit_std > 1e-10:
            best_z = (self.f_opt - fit_mean) / fit_std
            rank_reward = float(np.clip(0.5 + best_z / 4.0, 0.0, 1.0))
        else:
            rank_reward = 0.5

        # === Stagnation Detection ===
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        is_very_stagnant = self.stagnation_counter > self.max_stagnation

        # === Combine Rewards with Adaptive Weights ===
        if is_very_stagnant:
            # When very stuck, prioritize diversity and rank over raw improvement
            w_improve, w_spread, w_rank = 0.2, 0.5, 0.3
        elif is_stagnant:
            # When stagnating, balance all dimensions
            w_improve, w_spread, w_rank = 0.4, 0.3, 0.3
        else:
            # Normal operation: emphasize improvement
            w_improve, w_spread, w_rank = 0.6, 0.2, 0.2

        combined_reward = (w_improve * imp_reward + 
                          w_spread * spread_reward + 
                          w_rank * rank_reward)
        combined_reward = float(np.clip(combined_reward, 0.0, 1.0))

        # === Exploration Bonus ===
        if is_stagnant and spread_ratio < 0.1:
            combined_reward *= 0.5  # Penalize operators that don't help escape

        # === Update Sliding Windows ===
        op = self.current_operator
        self.operator_improvement_scores[op].append(imp_reward)
        self.operator_spread_scores[op].append(spread_reward)
        self.operator_rank_scores[op].append(rank_reward)

        window = self.reward_window_size
        for scores in [self.operator_improvement_scores, 
                       self.operator_spread_scores, 
                       self.operator_rank_scores]:
            if len(scores[op]) > window:
                scores[op].pop(0)

        # === Compute Aggregated Reward for Beta Update ===
        n = len(self.operator_improvement_scores[op])
        if n >= 3:
            # Use smoothed reward from window
            mean_improve = np.mean(self.operator_improvement_scores[op])
            mean_spread = np.mean(self.operator_spread_scores[op])
            mean_rank = np.mean(self.operator_rank_scores[op])
            
            # Weighted combination matching the adaptive weights
            smoothed_reward = (w_improve * mean_improve + 
                             w_spread * mean_spread + 
                             w_rank * mean_rank)
        elif n >= 1:
            smoothed_reward = combined_reward
        else:
            smoothed_reward = 0.5

        smoothed_reward = float(np.clip(smoothed_reward, 0.0, 1.0))

        # === Update Beta Distribution Parameters ===
        # More stable update: use running mean directly
        prior_strength = 2.0
        if n >= 1:
            # Bayesian-style update: blend prior with observed
            obs_weight = min(n / (n + 2.0), 0.8)
            effective_reward = (1 - obs_weight) * 0.5 + obs_weight * smoothed_reward
        else:
            effective_reward = smoothed_reward

        effective_reward = float(np.clip(effective_reward, 0.01, 0.99))

        # Update alpha/beta with controlled learning rate
        lr = 0.3
        self.alpha[op] = float(max(1.0, (1 - lr) * self.alpha[op] + lr * (1.0 + effective_reward * prior_strength)))
        self.beta[op] = float(max(1.0, (1 - lr) * self.beta[op] + lr * (1.0 + (1 - effective_reward) * prior_strength)))

        # Ensure reasonable bounds on Beta parameters
        self.alpha[op] = float(np.clip(self.alpha[op], 1.0, 50.0))
        self.beta[op] = float(np.clip(self.beta[op], 1.0, 50.0))
```