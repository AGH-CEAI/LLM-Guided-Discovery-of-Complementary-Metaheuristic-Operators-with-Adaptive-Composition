**Idea: InfoTheoretic Strategy Credit with Distribution Analysis**

Category C: Uses entropy of fitness distributions per strategy, mutual information between strategy selection and improvement events, and KL divergence to measure how each strategy shifts the population's fitness distribution. The method treats strategy selection as an information channel where "improvement signals" are transmitted.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """
    Information-theoretic credit assignment using:
    - Entropy of fitness distributions per strategy
    - Mutual information between strategy and improvement signal
    - KL divergence of strategy fitness from population baseline
    """
    n_strategies = len(self.strategy_scores)
    NP = len(strategy_used)
    
    if NP < 5:
        return
    
    # Initialize history tracking for distributional analysis
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    
    # Store current fitness for temporal analysis
    if hasattr(self, 'fitness'):
        self._fitness_history.append(self.fitness.copy())
        if len(self._fitness_history) > 20:
            self._fitness_history.pop(0)
    
    reward = np.zeros(n_strategies)
    
    # 1. Entropy-based reward: strategies that reduce fitness variance get positive credit
    for s in range(n_strategies):
        mask = strategy_used == s
        n_s = np.sum(mask)
        if n_s >= 3 and hasattr(self, 'fitness'):
            fitness_s = self.fitness[mask]
            # Compute entropy of binned fitness distribution
            n_bins = min(8, max(2, n_s // 2))
            bin_edges = np.percentile(fitness_s, np.linspace(0, 100, n_bins + 1))
            counts, _ = np.histogram(fitness_s, bins=bin_edges)
            probs = counts / n_s
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            # Lower entropy = more concentrated = potentially better exploitation
            reward[s] -= entropy * 0.3
    
    # 2. Mutual information: strategies with high MI with improvement signal
    n_improved = np.sum(improved)
    if 0 < n_improved < NP:
        # Build joint distribution P(strategy, improvement)
        p_joint = np.zeros((2, n_strategies))
        for s in range(n_strategies):
            mask = strategy_used == s
            n_s = np.sum(mask)
            if n_s > 0:
                n_imp_s = np.sum(improved[mask])
                p_joint[0, s] = n_imp_s / NP  # improved
                p_joint[1, s] = (n_s - n_imp_s) / NP  # not improved
        
        # Compute MI = sum_{i,s} P(i,s) * log(P(i,s) / (P(i)*P(s)))
        for s in range(n_strategies):
            p_s = np.sum(p_joint[:, s])
            if p_s > 1e-10:
                for i in range(2):
                    if p_joint[i, s] > 1e-10:
                        p_i = np.sum(p_joint[i, :])
                        mi_contrib = p_joint[i, s] * np.log(p_joint[i, s] / (p_i * p_s + 1e-12))
                        reward[s] += mi_contrib * 5.0  # Scale for numerical stability
    
    # 3. KL divergence: strategies shifting fitness distribution toward lower values
    if hasattr(self, 'fitness') and len(self._fitness_history) >= 2:
        prev_fitness = self._fitness_history[-2] if len(self._fitness_history) >= 2 else self.fitness
        for s in range(n_strategies):
            mask = strategy_used == s
            n_s = np.sum(mask)
            if n_s >= 2:
                current_s = self.fitness[mask]
                prev_s = prev_fitness[mask]
                # Normalize for distribution comparison
                if np.std(current_s) > 1e-10 and np.std(prev_s) > 1e-10:
                    curr_norm = (current_s - np.min(current_s) + 1e-6) / (np.ptp(current_s) + 1e-6)
                    prev_norm = (prev_s - np.min(prev_s) + 1e-6) / (np.ptp(prev_s) + 1e-6)
                    # KL(current || prev): lower current values shift distribution left
                    kl = np.mean(np.log(prev_norm + 1e-6) - np.log(curr_norm + 1e-6))
                    reward[s] += kl * 0.5
    
    # Apply momentum-based update with softmax normalization
    reward = np.clip(reward, -5, 5)
    reward = reward - np.mean(reward)  # Center rewards
    
    momentum = 0.7
    self.strategy_scores = momentum * self.strategy_scores + (1 - momentum) * reward
    
    # Softmax normalization with temperature
    temp = 1.5
    exp_scores = np.exp(self.strategy_scores / temp)
    self.strategy_scores = exp_scores / np.sum(exp_scores)
    
    # Ensure minimum probability to prevent strategy starvation
    self.strategy_scores = np.maximum(self.strategy_scores, 0.05)
    self.strategy_scores /= np.sum(self.strategy_scores)
```