**Idea: Rank-Correlation Landscape Adaptation**
One-line description: Use Spearman rank correlation between parent/trial fitness and rank-distribution statistics to infer landscape ruggedness and drive F/Cr adaptation (Category D: Fitness-landscape / rank-based).

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using fitness rank correlations and landscape ruggedness inference."""
    np_pop = len(improved_mask)
    
    # Compute rank correlation between parent fitness and trial fitness
    # This measures landscape ruggedness: high correlation = smooth, low = rugged
    if not hasattr(self, 'parent_fitness_history'):
        self.parent_fitness_history = []
        self.trial_fitness_history = []
        self.rank_correlation_history = []
        self.F_adapt_history = []
        self.Cr_adapt_history = []
    
    # Store recent fitness pairs for correlation analysis
    # Use the last generation's data (passed via improved_mask context)
    if hasattr(self, '_last_parent_fitness') and hasattr(self, '_last_trial_fitness'):
        parent_fit = self._last_parent_fitness
        trial_fit = self._last_trial_fitness
        
        # Compute Spearman rank correlation
        n = len(parent_fit)
        if n >= 5:
            parent_ranks = np.argsort(np.argsort(parent_fit))
            trial_ranks = np.argsort(np.argsort(trial_fit))
            
            # Spearman: correlation of ranks
            rank_diff = parent_ranks - trial_ranks
            spearman_rho = 1.0 - (6.0 * np.sum(rank_diff ** 2)) / (n * (n**2 - 1))
            spearman_rho = np.clip(spearman_rho, -1.0, 1.0)
            
            self.rank_correlation_history.append(spearman_rho)
            if len(self.rank_correlation_history) > 10:
                self.rank_correlation_history.pop(0)
        else:
            spearman_rho = 0.0
    else:
        spearman_rho = 0.0
    
    # Compute fitness rank distribution statistics
    if hasattr(self, '_last_fitness'):
        current_ranks = np.argsort(np.argsort(self._last_fitness))
        current_percentiles = current_ranks / (np_pop - 1) if np_pop > 1 else np.array([0.5])
        
        # Track if population ranks are converging (low spread = converging)
        rank_spread = np.ptp(current_percentiles)
        rank_entropy = -np.sum(current_percentiles * np.log(current_percentiles + 1e-10))
        
        if not hasattr(self, 'rank_entropy_history'):
            self.rank_entropy_history = []
        self.rank_entropy_history.append(rank_entropy)
        if len(self.rank_entropy_history) > 8:
            self.rank_entropy_history.pop(0)
        
        # Detect rank stagnation: entropy decreasing = population converging
        if len(self.rank_entropy_history) >= 3:
            entropy_trend = np.mean(np.diff(self.rank_entropy_history[-3:]))
        else:
            entropy_trend = 0.0
    else:
        rank_spread = 1.0
        entropy_trend = 0.0
    
    # Compute success rate from improved_mask
    success_rate = np.mean(improved_mask)
    
    # Initialize state on first call
    if not hasattr(self, 'rank_based_F'):
        self.rank_based_F = self.F_base
        self.rank_based_Cr = self.Cr_base
        self.rank_stagnation_counter = 0
        self.exploration_boost = 0.0
    
    # Analyze rank correlation to determine landscape regime
    if len(self.rank_correlation_history) >= 3:
        avg_rho = np.mean(self.rank_correlation_history[-3:])
        rho_trend = self.rank_correlation_history[-1] - self.rank_correlation_history[0]
    else:
        avg_rho = 0.0
        rho_trend = 0.0
    
    # Regime detection based on rank-based features
    landscape_smooth = avg_rho > 0.3
    landscape_rugged = avg_rho < -0.1
    ranks_converging = entropy_trend < -0.05
    ranks_stagnant = rank_spread < 0.3
    
    # Compute directional momentum from recent F adaptations
    self.F_adapt_history.append(F_used)
    self.Cr_adapt_history.append(Cr_used)
    if len(self.F_adapt_history) > 8:
        self.F_adapt_history.pop(0)
        self.Cr_adapt_history.pop(0)
    
    # Compute momentum: positive = F trending up, negative = F trending down
    if len(self.F_adapt_history) >= 3:
        F_recent = np.array(self.F_adapt_history[-3:])
        F_momentum = (F_recent[-1] - F_recent[0]) / (F_recent[0] + 1e-10)
    else:
        F_momentum = 0.0
    
    # Adaptation logic based on rank-correlation landscape analysis
    if landscape_smooth and success_rate > 0.2:
        # Smooth landscape: exploit with smaller F, moderate Cr
        # High rank correlation means offspring ranks predictable from parent ranks
        self.rank_based_F = 0.85 * self.rank_based_F + 0.15 * 0.4
        self.rank_based_Cr = 0.9 * self.rank_based_Cr + 0.1 * 0.6
        self.rank_stagnation_counter = max(0, self.rank_stagnation_counter - 1)
        self.exploration_boost *= 0.9
        
    elif landscape_rugged or (ranks_converging and not landscape_smooth):
        # Rugged landscape OR converging in complex terrain: explore with larger F
        # Low rank correlation means landscape has many local optima
        self.rank_based_F = 0.8 * self.rank_based_F + 0.2 * 0.9
        self.rank_based_Cr = 0.9 * self.rank_based_Cr + 0.1 * 0.2
        self.rank_stagnation_counter += 1
        self.exploration_boost = min(1.0, self.exploration_boost + 0.1)
        
    elif ranks_stagnant:
        # Population ranks stuck: need perturbation
        self.rank_based_F = 0.7 * self.rank_based_F + 0.3 * 1.0
        self.rank_based_Cr = 0.85 * self.rank_based_Cr + 0.15 * 0.8
        self.rank_stagnation_counter += 2
        self.exploration_boost = 1.0
        
    else:
        # Neutral regime: balance exploration/exploitation with momentum
        momentum_modulation = np.clip(F_momentum, 0.7, 1.3)
        self.rank_based_F = 0.9 * self.rank_based_F + 0.1 * (0.6 * momentum_modulation)
        self.rank_based_Cr = 0.95 * self.rank_based_Cr + 0.05 * 0.4
        self.rank_stagnation_counter = max(0, self.rank_stagnation_counter - 1)
        self.exploration_boost *= 0.95
    
    # Apply momentum-based override for consistent trends
    if abs(F_momentum) > 0.2 and len(self.F_adapt_history) >= 3:
        # If F has been consistently increasing/decreasing, amplify the trend
        if F_momentum > 0.2 and avg_rho > 0.1:
            # Exploitation working in smooth landscape: push F lower
            self.rank_based_F *= 0.9
        elif F_momentum < -0.2 and avg_rho < 0.0:
            # Exploration needed in rugged landscape: push F higher
            self.rank_based_F *= 1.15
    
    # Apply exploration boost for rugged landscapes
    if self.exploration_boost > 0.3:
        self.rank_based_F = self.rank_based_F * (1.0 + 0.2 * self.exploration_boost)
    
    # Clip to valid bounds
    self.F = np.clip(self.rank_based_F, 0.1, 2.0)
    self.Cr = np.clip(self.rank_based_Cr, 0.05, 0.95)
```