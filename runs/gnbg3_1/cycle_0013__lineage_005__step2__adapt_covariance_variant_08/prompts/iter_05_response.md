Looking at the task errors, the worst unsolved tasks (16-23) have errors in the 10^1 to 10^2 range, indicating severe multimodal/deceptive trapping where the algorithm gets stuck in local optima far from global solutions. The current variant_08 only perturbs the covariance matrix on stagnation — this is reactive and often too late.

**Idea: Forced Perturbation with Mean Reinitialization**
Instead of only perturbing the covariance matrix, this variant forces the mean position to jump to a diverse location in parameter space and dramatically increases step size when stagnation is detected. This creates a "kick" that actually relocates the search to a fundamentally different region, bypassing local optima by sampling from a completely new area.

```python
def _adapt_covariance_variant_08(self):
    """Forced perturbation with mean relocation for escaping deep local optima."""
    stagnation_threshold = max(30, self.dim * 2)
    is_stagnant = self.stagnation_counter > stagnation_threshold

    if is_stagnant:
        # Compute fitness-based weights for selecting a good restart point
        fit_min = np.min(self.fitness)
        fit_max = np.max(self.fitness)
        fit_range = max(fit_max - fit_min, 1e-10)
        weights_reloc = np.exp(-2.0 * (self.fitness - fit_min) / fit_range)
        weights_reloc /= np.sum(weights_reloc)
        
        # Select a promising but diverse candidate from population
        candidate_idx = np.random.choice(self.NP, p=weights_reloc)
        candidate = self.population[candidate_idx]
        
        # Compute population spread for perturbation scale
        pop_spread = np.mean(np.std(self.population, axis=0))
        perturb_scale = max(pop_spread * 3.0, (self.ub[0] - self.lb[0]) * 0.1)
        
        # Relocate mean to diverse region with forced exploration
        random_dir = np.random.randn(self.dim)
        random_dir /= (np.linalg.norm(random_dir) + 1e-10)
        self.mean = candidate + perturb_scale * random_dir
        
        # Clip to bounds
        self.mean = self._clip_to_bounds(self.mean)
        
        # Dramatically increase step size for global search
        self.sigma = min(self.sigma * 5.0, (self.ub[0] - self.lb[0]) * 0.2)
        
        # Reset covariance to moderate isotropic for fresh start
        eigvals = np.linalg.eigvalsh(self.C)
        avg_eig = np.mean(eigvals)
        self.C = np.eye(self.dim) * max(avg_eig, 1e-4)
        
        # Reset evolution path
        self.pc *= 0.1
        self.stagnation_counter = 0
    else:
        # Normal CMA-ES update
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C +
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)

        # Check condition number for numerical stability
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals_check) / np.min(eigvals_check)
        if cond > 1e7:
            self.C *= 0.5
```