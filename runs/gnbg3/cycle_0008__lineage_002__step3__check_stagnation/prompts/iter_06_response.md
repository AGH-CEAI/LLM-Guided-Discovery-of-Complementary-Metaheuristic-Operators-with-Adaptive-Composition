**Idea: Temporal Momentum & Autocorrelation Stagnation Detector**

This approach tracks the *rate of fitness improvement* and *autocorrelation* of the best fitness signal across generations. Rather than simple counter-based stagnation, it detects when improvement momentum has decayed and when the trajectory becomes temporally predictable (correlated) — both hallmarks of being trapped in a local basin. Uses exponential moving averages and derivative estimation to distinguish genuine stagnation from noisy plateaus.

```python
def _check_stagnation(self, best_fitness):
    """Detect stagnation via temporal dynamics: momentum decay and autocorrelation."""
    
    # Initialize temporal state
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
        self._ema_best = float('inf')
        self._ema_alpha = 0.2
        self._stagnation_counter = 0
        self._improvement_rates = []
        self._autocorr_buffer = []
        self._momentum = 0.0
        self._prev_fitness = float('inf')
    
    # Update EMA of best fitness (temporal smoothing)
    self._ema_best = (1 - self._ema_alpha) * self._ema_best + self._ema_alpha * best_fitness
    
    # Store history for autocorrelation and rate computation
    self._fitness_history.append(best_fitness)
    if len(self._fitness_history) > 50:
        self._fitness_history.pop(0)
    
    # Compute recent improvement rate (derivative approximation)
    if len(self._fitness_history) >= 5:
        recent_window = self._fitness_history[-5:]
        rate = (recent_window[0] - recent_window[-1]) / 4.0
        self._improvement_rates.append(rate)
        if len(self._improvement_rates) > 20:
            self._improvement_rates.pop(0)
        
        # Update momentum (EMA of rate)
        if len(self._improvement_rates) >= 2:
            self._momentum = 0.7 * self._momentum + 0.3 * rate
    
    # Compute autocorrelation at lag 1 (temporal predictability signal)
    if len(self._fitness_history) >= 10:
        hist = np.array(self._fitness_history)
        # Detrend by removing mean
        centered = hist - np.mean(hist)
        if np.std(centered) > 1e-12:
            autocorr_lag1 = np.corrcoef(centered[:-1], centered[1:])[0, 1]
        else:
            autocorr_lag1 = 0.0
        self._autocorr_buffer.append(autocorr_lag1)
        if len(self._autocorr_buffer) > 15:
            self._autocorr_buffer.pop(0)
    
    # Stagnation detection logic
    fitness_change = self._prev_fitness - best_fitness
    self._prev_fitness = best_fitness
    
    # Condition 1: No absolute improvement
    no_improvement = fitness_change <= 1e-10
    
    # Condition 2: Momentum decay (improvement rate approaching zero)
    momentum_stalled = False
    if len(self._improvement_rates) >= 5:
        recent_rates = self._improvement_rates[-5:]
        if all(abs(r) < 1e-6 for r in recent_rates):
            momentum_stalled = True
    
    # Condition 3: High autocorrelation (stuck in predictable trajectory)
    highly_correlated = False
    if len(self._autocorr_buffer) >= 5:
        recent_autocorr = self._autocorr_buffer[-5:]
        mean_autocorr = np.nanmean(recent_autocorr)
        if mean_autocorr > 0.7 and not np.isnan(mean_autocorr):
            highly_correlated = True
    
    # Condition 4: EMA has plateaued (smoothed signal not improving)
    ema_plateaued = False
    if len(self._fitness_history) >= 10:
        ema_window = self._fitness_history[-10:]
        if max(ema_window) - min(ema_window) < 1e-8:
            ema_plateaued = True
    
    # Increment counter only if multiple conditions hold
    if (no_improvement and momentum_stalled) or (highly_correlated and ema_plateaued):
        self._stagnation_counter += 1
    else:
        # Partial reset: reduce counter but don't fully reset
        self._stagnation_counter = max(0, self._stagnation_counter - 1)
    
    # Threshold for restart (adaptive based on population size)
    stagnation_threshold = max(15, self.NP // 2)
    
    return self._stagnation_counter >= stagnation_threshold
```