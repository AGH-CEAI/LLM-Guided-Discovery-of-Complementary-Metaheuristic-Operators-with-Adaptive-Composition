**Idea: Multi-Objective Thompson Sampling with Entropy-Maximizing Exploration**

A fundamentally different approach: instead of UCB's deterministic exploitation-biased selection, use Thompson Sampling with a multi-objective reward signal (combining success rate + diversity contribution + convergence momentum) and an entropy-maximizing exploration bonus to prevent premature commitment to a single operator. This directly targets the worst tasks (16, 19, 17, 23) which are stuck at ~1e+01 to 1e+03 — indicating the algorithm is trapped in local optima due to over-exploitation and lack of operator diversity.

```python
def _select_mutation_operator_batch(self):
        """Select mutation operator using Multi-Objective Thompson Sampling with entropy bonus."""
        total_counts = np.sum(self.operator_counts)
        
        # Compute multi-objective reward signal (3 components)
        avg_rewards = self.operator_rewards / np.maximum(self.operator_counts, 1)
        
        # Component 1: Success rate (exploitation)
        success_component = avg_rewards.copy()
        
        # Component 2: Diversity contribution (how much this operator explores different regions)
        diversity_component = np.zeros(self.n_strategies)
        if hasattr(self, 'operator_diversity') and len(self.operator_diversity) == self.n_strategies:
            diversity_component = self.operator_diversity / (np.maximum(np.sum(self.operator_diversity), 1e-10))
        else:
            # Fallback: favor less-used operators for diversity
            diversity_component = 1.0 / np.maximum(self.operator_counts, 1)
        
        # Component 3: Convergence momentum (recent improvement trend)
        momentum_component = np.zeros(self.n_strategies)
        if hasattr(self, 'operator_momentum') and len(self.operator_momentum) == self.n_strategies:
            momentum_component = np.clip(self.operator_momentum, 0, 1)
        
        # Combine components with adaptive weights
        combined_reward = (0.5 * success_component + 
                          0.3 * diversity_component + 
                          0.2 * momentum_component)
        
        # Prevent division by zero / NaN
        combined_reward = np.nan_to_num(combined_reward, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Thompson Sampling: sample from Gamma distribution parameterized by rewards
        # Shape = reward + 1 (avoid zero), Scale = 1
        shape_param = combined_reward + 1e-6
        sampled_scores = np.random.gamma(shape_param, 1.0)
        
        # Entropy-maximizing exploration bonus: favor operators with lower selection counts
        exploration_bonus = self.ucb_c * np.sqrt(np.log(total_counts + 1) / np.maximum(self.operator_counts, 1))
        exploration_bonus = np.clip(exploration_bonus, 0, 10.0)  # Prevent overflow
        
        # Final score with exploration bonus
        final_scores = sampled_scores + exploration_bonus
        
        # Select operator with highest final score
        selected = int(np.argmax(final_scores))
        
        # Initialize tracking arrays if needed
        if not hasattr(self, 'operator_diversity'):
            self.operator_diversity = np.ones(self.n_strategies)
        if not hasattr(self, 'operator_momentum'):
            self.operator_momentum = np.zeros(self.n_strategies)
        
        # Decay momentum (keep only recent history)
        self.operator_momentum *= 0.9
        
        # Update diversity tracking (will be used next call)
        self.operator_diversity *= 0.95
        
        return selected
```