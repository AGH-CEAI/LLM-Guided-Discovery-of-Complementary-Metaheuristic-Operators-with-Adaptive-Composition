Looking at the problem, I need to analyze why the worst tasks (16: 698.6, 19: 29.3, 20: 22.8, 23: 20.1, 18: 13.4) are stuck with errors 8+ decades above target. These large errors strongly suggest the optimizer is trapped in local optima or failing to explore the search space effectively.

**Key insight**: The current sampling draws ALL candidates from a single multivariate normal centered on `self.mean`. When trapped, this creates a collapsing search distribution that cannot escape. Tasks with large errors need aggressive diversification.

**My strategy**: Implement **archive-based restart sampling** — maintain a persistent archive of top solutions from previous restarts, and when stagnation triggers a restart, sample a FRACTION of the new population FROM THIS ARCHIVE. This forces exploration of diverse basins while preserving knowledge of good solutions.

This is fundamentally different from: (1) original which never uses historical solutions for sampling, and (2) variant_01_idea_0 which performed worse overall — likely by disrupting the core CMA mechanism rather than augmenting it with auxiliary exploration.

**Idea: Archive-Restart Sampling**
When stagnating, reinitialize mean far from current best but SAMPLE FROM ARCHIVE to maintain diversity, never discarding good solutions.
```python
def _sample_population_batch(self):
        """
        Sample a batch of candidates with archive-based restart diversification.
        Maintains an archive of best solutions across restarts to prevent getting
        trapped in local optima by sampling from diverse historical solutions.
        """
        # Initialize archive if first call
        if not hasattr(self, 'archive') or self.archive is None:
            self.archive = []
        
        # Check if we should force restart diversity sampling
        force_archive_sampling = (self.stagnation_counter > self.dim)
        
        z = np.random.randn(self.pop_size, self.dim)
        
        # If using archive sampling, bias toward diverse historical solutions
        if force_archive_sampling and len(self.archive) > 0:
            archive_size = len(self.archive)
            # Sample a portion from archive (proportional to stagnation severity)
            archive_portion = min(len(self.archive), max(1, self.pop_size // 2))
            archive_indices = np.random.choice(archive_size, archive_portion, replace=False)
            
            for i, idx in enumerate(archive_indices[:archive_portion]):
                if i < self.pop_size:
                    # Sample from neighborhood of archived solution
                    archive_solution = np.array(self.archive[idx])
                    # Use larger variance for exploration
                    z[i] = np.random.randn(self.dim) * 3.0
        
        # Ensure D and B are valid
        if self.D is None or self.B is None or np.any(self.D <= 0):
            self.D = np.ones(self.dim)
            self.B = np.eye(self.dim)
        
        # Transform: x = mean + sigma * B * D * z
        safe_D = np.maximum(self.D, 1e-20)
        scaled = z * safe_D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        
        # Add archived solutions directly to population
        if len(self.archive) > 0 and self.generation % 3 == 0:
            inject_count = min(len(self.archive), max(1, self.pop_size // 4))
            archive_samples = np.array(
                list(self.archive[:inject_count]) + 
                [np.random.randn(self.dim) for _ in range(inject_count)]
            )[:inject_count]
            
            if archive_samples.shape[0] >= inject_count:
                start_idx = max(0, self.pop_size - inject_count)
                for i in range(min(inject_count, self.pop_size - start_idx)):
                    if start_idx + i < self.pop_size:
                        population[start_idx + i] = archive_samples[i]
        
        return population, z
```