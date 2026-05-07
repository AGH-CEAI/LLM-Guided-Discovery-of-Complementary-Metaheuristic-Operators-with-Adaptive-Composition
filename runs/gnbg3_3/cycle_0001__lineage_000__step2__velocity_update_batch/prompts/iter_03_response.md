**Idea: Entropy-Modulated Velocity with Gaussian Distribution Fitting**

Category C: Information-theoretic / distributional. Fit a multivariate Gaussian to the population, compute its entropy, and use entropy-based scaling to modulate exploration vs exploitation — low entropy (population collapsed) triggers aggressive dispersive velocity components.

```python
def _velocity_update_batch(self):
    """Update velocities with entropy-modulated distribution-guided exploration."""
    cognitive, social = self._adaptive_coefficients()
    
    # Generate random matrices once for efficiency
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component: attraction to personal best
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # Social component: attraction to local best (ring topology)
    social_component = social * r2 * (self.local_best - self.population)
    
    # --- INFORMATION-THEORETIC COMPONENT (Category C) ---
    # Fit a multivariate Gaussian to the current population
    mean_pop = np.mean(self.population, axis=0)
    centered = self.population - mean_pop
    
    # Compute population covariance with regularization for numerical stability
    cov_pop = np.cov(centered.T)
    cov_pop += np.eye(self.dim) * 1e-8  # Regularization
    
    try:
        # Eigendecomposition of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_pop)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)  # Ensure positive
        
        # Compute differential entropy of the Gaussian: H = 0.5 * log((2πe)^d * det(cov))
        # Entropy increases with spread/determinant of covariance
        log_det = np.sum(np.log(eigenvalues))
        entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + log_det)
        
        # Normalize entropy to [0, 1] relative to expected range for this dimension
        # Max entropy for uniform distribution in range [-100, 100] ≈ dim * log(range)
        max_entropy = self.dim * np.log(self.upper_bound - self.lower_bound + 1e-10)
        min_entropy = self.dim * np.log(1e-6)  # Highly concentrated
        entropy_normalized = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0, 1)
        
        # Entropy-based dispersive force: when entropy is LOW, push particles toward mean
        # This combats premature convergence detected via information-theoretic measure
        dispersive_strength = 0.5 * (1.0 - entropy_normalized)
        
        # Compute Mahalanobis-like distance for each particle from distribution center
        # Use inverse square root of eigenvalues for scaling
        inv_sqrt_eigen = 1.0 / np.sqrt(eigenvalues + 1e-10)
        scaled_diff = centered * inv_sqrt_eigen  # Broadcast: (np, dim)
        mahal_dist = np.linalg.norm(scaled_diff, axis=1, keepdims=True)  # (np, 1)
        
        # Normalize distances to [0, 1] per generation
        max_mahal = np.max(mahal_dist) + 1e-10
        mahal_normalized = mahal_dist / max_mahal
        
        # Direction toward mean (normalized)
        to_mean = mean_pop - self.population  # (np, dim)
        to_mean_norm = np.linalg.norm(to_mean, axis=1, keepdims=True) + 1e-10
        to_mean_dir = to_mean / to_mean_norm
        
        # Dispersive velocity component: push toward mean, modulated by entropy and distance
        # Particles far from center get pushed more; particles near center get less push
        dispersive_component = (
            dispersive_strength *
            mahal_normalized *
            to_mean_dir *
            np.random.uniform(0, 1, (self.np, self.dim))
        )
        
        # Also add a small random perturbation scaled by inverse entropy
        # (more random exploration when population is collapsed)
        random_explore = 0.3 * (1.0 - entropy_normalized) * np.random.uniform(-1, 1, (self.np, self.dim))
        
        entropy_velocity = dispersive_component + random_explore
        
    except np.linalg.LinAlgError:
        # Fallback: small random exploration on decomposition failure
        entropy_velocity = 0.1 * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # Velocity update with inertia, cognitive, social, mutation, and entropy-based components
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        entropy_velocity
    )
    
    # Clip velocity to bounds
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```