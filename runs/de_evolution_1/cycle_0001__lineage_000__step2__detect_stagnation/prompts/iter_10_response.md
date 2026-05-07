Looking at the results, I notice that variant_09 tends to do best on the harder tasks (10-23), while the original and variant_05 do best on easier tasks (0-9). The key insight is that ALL tasks remain unsolved - even task 0 is still at 1e-07, one decade from target.

The core problem: the current stagnation detection triggers restarts too aggressively (IPOP doubles pop size, wasting budget) OR too conservatively (not restarting when stuck in local optima). For high-dimensional multimodal tasks (16, 19, 20, etc.), we need many restarts with small populations to explore diverse basins. For near-solved tasks (0, 1, 6), we need to NOT restart during fine-tuning.

My approach: **ultra-aggressive stagnation detection** that triggers restarts very quickly (after just a few generations without improvement), combined with tracking the relative improvement rate. This forces many more restarts with smaller populations, maximizing exploration of the search space. For tasks already close to the optimum, the fitness-relative criterion prevents premature restarts during fine convergence. The key differentiator is using a fitness-proportional stagnation limit — when the best fitness is large (stuck far from optimum), restart almost immediately; when close to zero, allow more patience.

**Idea: Fitness-Proportional Ultra-Aggressive Restart**
Stagnation limit scales with log of best fitness — large errors trigger near-immediate restarts for maximum exploration, small errors allow patient convergence.

```python
def _detect_stagnation(self):
    """Fitness-proportional stagnation: restart faster when far from optimum."""
    # Base stagnation limit is very short to encourage many restarts
    base_limit = 3 + int(5 * self.dim / self.pop_size)
    
    # Scale patience by how good our current best fitness is
    # If best_fitness is large (>>1), we're far from optimum -> restart quickly
    # If best_fitness is tiny (<1e-6), we might be converging -> be patient
    if self.best_fitness < np.inf and self.best_fitness > 0:
        log_fitness = np.log10(max(self.best_fitness, 1e-20))
        # log_fitness ranges from ~-20 (excellent) to ~3 (terrible)
        # Map to patience multiplier: terrible -> 0.5x (very impatient), excellent -> 5x (patient)
        if log_fitness > 0:
            # Far from optimum: be very impatient
            patience_mult = max(0.3, 1.0 / (1.0 + log_fitness))
        else:
            # Close to optimum: be more patient
            patience_mult = min(8.0, 1.0 + abs(log_fitness) * 0.3)
    else:
        patience_mult = 1.0
    
    stag_limit = max(2, int(base_limit * patience_mult))
    
    if self.stagnation_counter > stag_limit:
        return True
    
    # Sigma collapsed - definitely restart
    if self.sigma < 1e-16:
        return True
    
    # Sigma too large relative to domain (exploration has diverged)
    if self.sigma > 1e4:
        return True
    
    # Condition number too high
    if np.max(self.D) / max(np.min(self.D), 1e-30) > 1e6:
        return True
    
    # Check if fitness history shows flatness (no relative improvement)
    if hasattr(self, 'best_fitness_history') and len(self.best_fitness_history) > 0:
        self.best_fitness_history.append(self.best_fitness)
        # Keep only recent history
        if len(self.best_fitness_history) > 50:
            self.best_fitness_history = self.best_fitness_history[-50:]
        
        window = min(20, len(self.best_fitness_history))
        if window >= 5:
            recent = self.best_fitness_history[-window:]
            old_val = recent[0]
            new_val = recent[-1]
            # If relative improvement over window is negligible
            if old_val > 0 and abs(old_val - new_val) / (abs(old_val) + 1e-30) < 1e-12:
                return True
    else:
        self.best_fitness_history = [self.best_fitness]
    
    # Detect if eigenvalues of C are all nearly equal but sigma is tiny
    # (converged to wrong spot)
    if self.sigma < 0.01 and self.best_fitness > 1.0:
        return True
    
    return False
```