**Idea: Deme-Based Niching Sampling**
Sample from multiple spatially-distributed demes (subpopulations) to maintain diversity and escape local optima on multi-modal landscapes.

```python
def _sample_population_batch(self):
    """
    Sample a batch using deme-based niching: maintain multiple subpopulations
    distributed across the search space to preserve diversity on multi-modal tasks.
    Returns array of shape (pop_size, dim).
    """
    pop_size = self.pop_size
    dim = self.dim
    
    # Number of demes: sqrt(pop_size) demes, each with sqrt(pop_size) individuals
    n_demes = max(2, int(np.sqrt(pop_size)))
    deme_size = pop_size // n_demes
    
    all_candidates = []
    all_z = []
    
    # Compute current diversity: ratio of max to min eigenvalue
    eig_ratio = np.max(self.D) / np.min(self.D) if np.min(self.D) > 1e-20 else 1e7
    
    for d in range(n_demes):
        # Each deme has its own center distributed along principal axes
        # Spread demes along the dominant eigenvector to explore elongated valleys
        if eig_ratio > 10.0:
            # Ill-conditioned: spread demes along the major axis
            axis_weight = (d - n_demes / 2) / max(n_demes / 2, 1)
            major_axis = self.B[:, -1]  # eigenvector with largest eigenvalue
            deme_center = self.mean + axis_weight * 2.0 * self.sigma * self.D[-1] * major_axis
        else:
            # Isotropic: spread demes radially from mean
            angle = 2.0 * np.pi * d / n_demes
            spread_radius = self.sigma * min(5.0, np.mean(self.D))
            deme_center = self.mean + spread_radius * np.concatenate([
                [np.cos(angle)],
                [np.sin(angle)] + [0.0] * (dim - 2)
            ])
        
        # Sample individuals around this deme center
        z_deme = np.random.randn(deme_size, dim)
        scaled = z_deme * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        deme_pop = deme_center[np.newaxis, :] + self.sigma * 0.5 * rotated
        all_candidates.append(deme_pop)
        all_z.append(z_deme)
    
    # Handle remainder (if pop_size not divisible by n_demes)
    remainder = pop_size - len(all_candidates) * deme_size
    if remainder > 0:
        z_rem = np.random.randn(remainder, dim)
        scaled = z_rem * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        rem_pop = self.mean[np.newaxis, :] + self.sigma * rotated
        all_candidates.append(rem_pop)
        all_z.append(z_rem)
    
    # Concatenate all demes
    population = np.concatenate(all_candidates, axis=0)
    z_vectors = np.concatenate(all_z, axis=0)
    
    return population, z_vectors
```