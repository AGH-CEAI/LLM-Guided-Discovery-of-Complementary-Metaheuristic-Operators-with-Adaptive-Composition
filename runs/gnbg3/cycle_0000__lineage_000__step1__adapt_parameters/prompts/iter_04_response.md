**Idea: Fitness-Rank Percentile Steering with Success-Weighted Memory**

Adapt F and CR by tracking fitness rank percentiles and maintaining a success-weighted memory of which parameter regions worked historically. Use improvement_rate to distinguish exploration (low improvement) vs exploitation (high improvement) regimes, then steer F upward for exploration and CR toward 1.0 when fitness ranks become concentrated (indicating convergence pressure).

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using fitness rank percentiles and success-history memory.
    
    Category D: Fitness-landscape / rank-based reasoning only.
    Uses fitness percentiles, rank concentration, and success memory — NOT distances.
    """
    # Compute fitness rank percentiles (population-level fitness landscape signal)
    valid_fitness = self.fitness[np.isfinite(self.fitness)]
    if len(valid_fitness) < 2:
        return
    
    # Normalize fitness to ranks (0=best, 1=worst)
    ranks = np.argsort(np.argsort(valid_fitness)) / (len(valid_fitness) - 1)
    
    # Fitness landscape signals
    rank_spread = np.percentile(ranks, 90) - np.percentile(ranks, 10)  # rank concentration
    rank_median = np.median(ranks)  # where is the "center of mass"?
    rank_tail_ratio = np.mean(ranks > 0.8)  # fraction in poor fitness tail
    
    # Regime detection based on improvement rate
    if improvement_rate < 0.05:
        regime = "stagnation"
        regime_factor = 1.4  # boost exploration
    elif improvement_rate < 0.15:
        regime = "exploration"
        regime_factor = 1.2
    elif improvement_rate < 0.35:
        regime = "balanced"
        regime_factor = 1.0
    else:
        regime = "exploitation"
        regime_factor = 0.85
    
    # Success-history memory: maintain sliding window of (fitness_state, F, CR, success)
    if not hasattr(self, '_param_memory'):
        self._param_memory = []
        self._memory_max = 20
    
    # Encode current fitness state as tuple
    fitness_state = (round(rank_spread, 2), round(rank_median, 2), regime)
    
    # Record current parameters and their outcome
    self._param_memory.append((fitness_state, self.F, self.CR, improvement_rate))
    if len(self._param_memory) > self._memory_max:
        self._param_memory.pop(0)
    
    # Query memory: find similar fitness states and their successful parameters
    similar_states = [(s, ir) for (s, _, _, ir) in self._param_memory 
                      if s[0] == fitness_state[0] and s[2] == regime]
    
    # Adaptive F: base on regime, nudge toward historically successful F
    if len(similar_states) >= 3:
        avg_success_ir = np.mean([ir for (_, ir) in similar_states])
        if avg_success_ir > improvement_rate:
            # Similar states worked better — move toward those F values
            historical_fs = [f for (s, f, _, ir) in self._param_memory 
                            if s[0] == fitness_state[0] and s[2] == regime]
            target_F = np.mean(historical_fs)
            self.F = self.F * 0.7 + target_F * 0.3
        else:
            self.F = self.F * regime_factor
    else:
        # No history — use regime-based adjustment
        if regime == "stagnation":
            self.F = min(self.F * 1.3, 1.5)  # aggressive exploration
        elif regime == "exploration":
            self.F = min(self.F * 1.15, 1.2)
        elif regime == "exploitation":
            self.F = max(self.F * 0.9, 0.3)  # fine-tuning
    
    # Adaptive CR: based on rank concentration (not distance)
    # High concentration = population converging = increase CR for convergence pressure
    # Low concentration = diverse landscape = decrease CR for exploration
    if rank_spread < 0.3:
        # Tight fitness cluster — use higher CR to converge faster
        target_CR = 0.92
    elif rank_spread > 0.6:
        # Wide fitness spread — use lower CR to explore diverse regions
        target_CR = 0.6
    else:
        target_CR = 0.8
    
    # Blend toward target based on regime
    self.CR = self.CR * 0.6 + target_CR * 0.4
    self.CR = np.clip(self.CR, 0.3, 0.98)
    self.F = np.clip(self.F, 0.2, 1.6)
    
    # Record history for temporal analysis
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
```