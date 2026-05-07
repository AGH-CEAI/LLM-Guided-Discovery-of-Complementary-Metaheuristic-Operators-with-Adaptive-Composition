**Idea: SHADE-style Global Mean Adaptation**
A fundamentally different approach using SHADE's success-history mechanism with weighted Lehmer mean for F and CR, computing global parameters from successful individuals rather than per-individual JADE-style updates.

```python
def _adapt_F_CR_jade_style(self, population, fitness, trials, trial_fitness, improved_mask):
    if len(trial_fitness) < len(trials):
        trials = trials[:len(trial_fitness)]
    
    # Track successful F values using Lehmer mean (SHADE-style)
    successful_F = []
    successful_CR = []
    
    for i in range(min(len(trials), self.NP)):
        if improved_mask[i]:
            # Compute F_i from difference vector (like JADE)
            diff = np.abs(trials[i] - population[i])
            norm_diff = np.linalg.norm(diff)
            if norm_diff > 1e-10:
                fi = np.abs(trials[i] - population[i]) / (norm_diff + 1e-10)
                successful_F.append(np.mean(fi))
            else:
                successful_F.append(self.rng.uniform(0.5, 0.9))
            successful_CR.append(self.CR[i])
    
    # Update archive with improved solutions
    improved_trials = trials[improved_mask]
    if len(improved_trials) > 0:
        self.archive.extend(improved_trials.tolist())
        if len(self.archive) > self.archive_max_size:
            remove_n = len(self.archive) - self.archive_max_size
            self.archive = self.archive[remove_n:]
    
    # SHADE-style weighted Lehmer mean for global F and CR
    if len(successful_F) > 0:
        f_weights = np.array(successful_F) ** 2
        f_weights = f_weights / (np.sum(f_weights) + 1e-10)
        mean_F = float(np.sum(f_weights * np.array(successful_F)))
        mean_F = np.clip(mean_F, 0.1, 1.0)
        
        cr_arr = np.array(successful_CR)
        cr_weights = cr_arr ** 2 + 1e-10
        cr_weights = cr_weights / (np.sum(cr_weights) + 1e-10)
        mean_CR = float(np.sum(cr_weights * cr_arr))
        mean_CR = np.clip(mean_CR, 0.0, 1.0)
        
        # Store in memory buffers (circular)
        if not hasattr(self, 'F_memory'):
            self.F_memory = np.full(100, 0.5)
            self.CR_memory = np.full(100, 0.5)
            self.memory_idx = 0
        
        self.F_memory[self.memory_idx] = mean_F
        self.CR_memory[self.memory_idx] = mean_CR
        self.memory_idx = (self.memory_idx + 1) % len(self.F_memory)
    
    # Assign F and CR to all individuals from memory
    if hasattr(self, 'F_memory'):
        valid_F = self.F_memory[self.F_memory > 0.1]
        valid_CR = self.CR_memory[(self.CR_memory > 0.0) & (self.CR_memory < 1.0)]
        
        if len(valid_F) > 0:
            for i in range(self.NP):
                self.F[i] = float(np.clip(
                    valid_F[self.rng.integers(0, len(valid_F))] * self.rng.uniform(0.9, 1.1),
                    0.1, 1.0
                ))
        
        if len(valid_CR) > 0:
            for i in range(self.NP):
                self.CR[i] = float(np.clip(
                    valid_CR[self.rng.integers(0, len(valid_CR))] * self.rng.uniform(0.9, 1.1),
                    0.0, 1.0
                ))
        else:
            self.CR[:] = np.clip(self.CR[:], 0.0, 1.0)
    else:
        self.CR[:] = np.clip(self.CR[:], 0.0, 1.0)
```