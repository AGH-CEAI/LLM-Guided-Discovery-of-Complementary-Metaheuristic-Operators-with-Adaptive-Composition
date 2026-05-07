Looking at the uncovered tasks (1, 8, 18, 20, 22), I notice they share a critical pattern: **every single variant makes them worse than the original**. This suggests these tasks have variable dependencies or epistasis that existing operators (which treat dimensions independently) are breaking. The block crossover helps overall but still hurts these tasks.

**Analysis:**
1. All variants use per-dimension independent crossover decisions
2. The uncovered tasks likely have variable linkage patterns where groups of variables should be inherited together
3. We need an operator that learns which dimensions tend to co-evolve during successful optimization and preserves those groupings

**Idea: Linkage-Aware Crossover**
This operator maintains a symmetric linkage matrix tracking which dimensions tend to be modified together during successful improvements. When doing crossover, it uses this matrix to bias decisions toward keeping linked dimensions together, preserving epistatic relationships.

```python
def _crossover_block(self, population, trial_population):
    """Op 3: variant_10 - Linkage-aware cluster crossover."""
    cr_batch = self.CR.copy()
    n_trials = self.NP
    dim = self.dim

    # Initialize or validate linkage matrix
    if not hasattr(self, '_linkage_matrix') or self._linkage_matrix is None:
        self._linkage_matrix = np.ones((dim, dim))
    elif self._linkage_matrix.shape[0] != dim:
        self._linkage_matrix = np.ones((dim, dim))

    # Update linkage: track which dimensions change together in trials
    diff = trial_population - population[:n_trials]
    changed = np.abs(diff) > 1e-8
    for d1 in range(dim):
        for d2 in range(dim):
            if d1 != d2:
                # Variables that change together get stronger linkage
                self._linkage_matrix[d1, d2] += 0.005 * (changed[:, d1] & changed[:, d2]).sum()

    # Decay old linkage to avoid lock-in
    self._linkage_matrix *= 0.9999
    np.fill_diagonal(self._linkage_matrix, 1.0)

    # Normalize linkage to get co-inheritance probabilities
    row_sums = self._linkage_matrix.sum(axis=1, keepdims=True)
    linkage_prob = self._linkage_matrix / (row_sums + 1e-10)

    # Build crossover mask: use CR and linkage to select dimensions
    cross_mask = np.zeros((n_trials, dim), dtype=bool)
    for i in range(n_trials):
        # Start with CR-based probability
        base_prob = cr_batch[i] * np.ones(dim)
        # Blend with linkage probability (favor inheriting linked dims together)
        blend_prob = 0.6 * base_prob + 0.4 * linkage_prob[i]
        blend_prob = np.clip(blend_prob, 0.0, 0.99)
        cross_mask[i] = self.rng.uniform(size=dim) < blend_prob[i]

    # Ensure at least one dimension is crossed
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        if not cross_mask[i, :].any():
            cross_mask[i, j_rand[i]] = True

    offspring = np.where(cross_mask, trial_population, population[:n_trials])
    return offspring
```