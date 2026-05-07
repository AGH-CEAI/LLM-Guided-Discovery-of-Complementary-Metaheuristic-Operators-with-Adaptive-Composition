**Idea: Context-Aware Thompson Sampling with Problem Structure Features**

This approach extracts real-time features from the optimization state (condition number, diversity ratio, fitness scale, convergence phase) and uses them to bias Thompson Sampling toward operators that historically performed well under similar problem structures. Unlike standard Thompson Sampling that only uses accumulated rewards, this method creates a context-aware mapping from problem characteristics to operator preferences.

```python
def _select_operator_thompson(self):
    """Context-aware Thompson Sampling using problem structure features."""
    # Extract meta-features describing current problem state
    fitness_scale = max(abs(self.f_opt), 1.0)
    fitness_scale_log = np.log1p(fitness_scale)
    
    pc_norm = float(np.linalg.norm(self.pc))
    
    eigvals = np.linalg.eigvalsh(self.C)
    cond = float(np.max(eigvals) / max(float(np.min(eigvals)), 1e-10))
    cond_log = np.log1p(cond)
    
    pop_diversity = float(np.mean(np.std(self.population, axis=0)))
    expected_range = float(self.ub[0] - self.lb[0])
    diversity_ratio = float(np.clip(pop_diversity / (expected_range + 1e-10), 0.0, 1.0))
    
    improvement = float(getattr(self, 'improvement_ema', 0.0))
    
    gen = float(self.generation)
    phase = float(np.clip(gen / max(1.0, float(self.max_stagnation)), 0.0, 1.0))
    
    meta_features = np.array([fitness_scale_log, pc_norm, cond_log, diversity_ratio, improvement, phase])
    
    # Initialize context tracking
    if not hasattr(self, 'context_history'):
        self.context_history = {op: [] for op in range(self.num_operators)}
        self.context_rewards = {op: [] for op in range(self.num_operators)}
    
    # Compute context-similarity weighted reward for each operator
    operator_context_scores = np.zeros(self.num_operators)
    
    for op in range(self.num_operators):
        if len(self.context_history[op]) >= 3:
            contexts = np.array(self.context_history[op])
            rewards = np.array(self.context_rewards[op])
            
            # Gaussian kernel similarity to current context
            diffs = contexts - meta_features
            # Normalize by feature ranges for fair comparison
            feature_ranges = np.array([5.0, 10.0, 10.0, 0.5, 5.0, 1.0])
            similarities = np.exp(-np.sum((diffs / (feature_ranges + 1e-10))**2, axis=1) * 0.5)
            
            # Weight by recency (exponential decay)
            n = len(rewards)
            recency_weights = np.array([0.5 ** ((n - 1 - i) / 5.0) for i in range(n)])
            
            weighted_rewards = similarities * recency_weights * np.maximum(rewards, 0.0)
            operator_context_scores[op] = float(np.sum(weighted_rewards))
    
    # Combine standard Thompson sampling with context-aware scores
    # Context scores provide bias toward historically successful operators for similar problem states
    context_boost = np.zeros(self.num_operators)
    max_score = float(np.max(operator_context_scores))
    if max_score > 1e-10:
        context_boost = operator_context_scores / (max_score + 1e-10) * 2.0
    
    # Apply boost to Beta distribution parameters
    boosted_alpha = self.alpha + context_boost
    boosted_beta = self.beta.copy()
    
    # Thompson sampling from boosted distributions
    samples = np.random.beta(boosted_alpha, boosted_beta)
    self.current_operator = int(np.argmax(samples))
    self.operator_counts[self.current_operator] += 1
    
    # Record current context and reward for future decisions
    op = self.current_operator
    reward = float(max(0.0, getattr(self, 'f_opt_prev', self.f_opt) - self.f_opt))
    
    self.context_history[op].append(meta_features)
    self.context_rewards[op].append(reward)
    
    max_contexts = 50
    if len(self.context_history[op]) > max_contexts:
        self.context_history[op] = self.context_history[op][-max_contexts:]
        self.context_rewards[op] = self.context_rewards[op][-max_contexts:]
    
    return self.current_operator
```