**Idea: Rank-Based Multi-Component Credit Assignment**
Replaces log-scaled improvement with a composite reward combining rank-based selection quality, diversity maintenance, and regret relative to random sampling. This handles deceptive problems where operators get stuck in local optima and produce zero improvement reward.

```python
def _update_operator_rewards(self):
        """Rank-based multi-component credit assignment for deceptive multi-modal optimization."""
        # === Initialize tracking state ===
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_selection_quality'):
            self.operator_selection_quality = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_diversity_contribution'):
            self.operator_diversity_contribution = np.zeros(self.num_operators)
        
        # === Component 1: Rank-based selection quality ===
        # Rank each trial candidate by fitness (lower fitness = better rank = higher reward)
        if len(self.trial_fitness) >= 2:
            ranks = np.argsort(np.argsort(self.trial_fitness))  # 0 = best, N-1 = worst
            n_trials = len(self.trial_fitness)
            rank_fraction = ranks / max(n_trials - 1, 1)  # Normalize to [0, 1]
            avg_rank_fraction = float(np.mean(rank_fraction))
            # Invert: lower rank_fraction is better -> higher reward
            selection_quality = 1.0 - avg_rank_fraction
        else:
            selection_quality = 0.5
        
        # === Component 2: Diversity maintenance contribution ===
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Individual diversity contribution via pairwise distances
        if len(self.population) >= 2:
            pop_center = np.mean(self.population, axis=0)
            dists = np.linalg.norm(self.population - pop_center, axis=1)
            diversity_contribution = float(np.mean(dists) / (expected_var ** 0.5 + 1e-10))
            diversity_contribution = np.clip(diversity_contribution, 0.0, 1.0)
        else:
            diversity_contribution = 0.5
        
        # === Component 3: Regret-based (comparing to expected random performance) ===
        # Compute expected fitness from random sampling as baseline
        if not hasattr(self, 'random_baseline_fitness'):
            self.random_baseline_fitness = float(np.mean(self.func(np.random.uniform(
                self.lb, self.ub, (min(1000, 10 * self.dim), self.dim))))
            )
        if not hasattr(self, 'regret_ema'):
            self.regret_ema = max(0.0, self.random_baseline_fitness - self.f_opt)
        
        # Regret: gap between current best and what random sampling would achieve
        regret = max(0.0, self.random_baseline_fitness - self.f_opt)
        self.regret_ema = 0.9 * self.regret_ema + 0.1 * regret
        # Higher regret = worse = operators not doing well
        regret_score = 1.0 / (1.0 + np.log1p(max(self.regret_ema, 1e-10)))
        
        # === Component 4: Stagnation-aware exploration bonus ===
        is_stagnant = self.stagnation_counter > self.max_stagnation // 3
        stagnation_bonus = 2.0 if is_stagnant else 1.0
        
        # === Combine components with adaptive weights ===
        # Weight by stagnation: when stagnant, emphasize diversity
        if is_stagnant:
            reward = (0.2 * selection_quality + 0.6 * diversity_contribution + 0.2 * regret_score)
        else:
            reward = (0.5 * selection_quality + 0.3 * diversity_contribution + 0.2 * regret_score)
        
        reward *= stagnation_bonus
        reward = float(np.clip(reward, 0.0, 5.0))
        
        # === Update per-operator tracking ===
        op = self.current_operator
        self.operator_cumulative_reward[op] += reward
        self.operator_selection_quality[op] = 0.9 * self.operator_selection_quality[op] + 0.1 * selection_quality
        self.operator_diversity_contribution[op] = 0.9 * self.operator_diversity_contribution[op] + 0.1 * diversity_contribution
        
        # === Compute normalized reward for Beta distribution update ===
        total_reward = float(np.sum(self.operator_cumulative_reward))
        if total_reward > 1e-10:
            normalized_reward = self.operator_cumulative_reward[op] / total_reward
        else:
            normalized_reward = 1.0 / self.num_operators
        
        # Also factor in selection quality and diversity contribution
        composite_score = 0.5 * normalized_reward + 0.3 * self.operator_selection_quality[op] + 0.2 * self.operator_diversity_contribution[op]
        composite_score = float(np.clip(composite_score, 0.0, 1.0))
        
        # === Update sliding window ===
        self.operator_rewards[op].append(composite_score)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # === Update Beta distribution parameters for Thompson Sampling ===
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            # Bayesian update with uncertainty
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1.0 - mean_reward) * ((n - 1) * (1.0 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            self.alpha[op] = 1.0 + sum_reward
            self.beta[op] = 1.0 + (n - sum_reward)
```