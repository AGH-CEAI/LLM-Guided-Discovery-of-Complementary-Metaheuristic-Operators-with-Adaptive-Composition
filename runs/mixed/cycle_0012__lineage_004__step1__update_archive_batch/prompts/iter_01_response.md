**Idea: Hypergrid-Diversity Archive**
A spatial diversity-preserving archive that divides the search space into a hypergrid and maintains only the best solution per cell, ensuring even coverage of the search space rather than pure fitness-based selection.

```python
def _update_archive_batch(self, archive, new_solutions, new_fitness):
    if len(new_solutions) == 0:
        return archive
    
    # Hypergrid parameters
    n_cells = max(2, int(np.sqrt(self.archive_max)))
    cell_capacity = max(1, self.archive_max // (n_cells ** self.dim) if self.dim <= 5 else 1)
    
    # Normalize function for grid mapping
    def normalize(solutions):
        range_val = self.upper - self.lower
        if range_val == 0:
            return np.zeros_like(solutions)
        return (solutions - self.lower) / range_val
    
    # Get current archive solutions for comparison
    current_sols = archive
    current_fits = np.array([-np.inf] * len(archive)) if len(archive) > 0 else np.array([])
    
    # Combine all candidates
    all_solutions = np.vstack([current_sols, new_solutions]) if len(current_sols) > 0 else new_solutions
    all_fitness = np.concatenate([current_fits, new_fitness]) if len(current_fits) > 0 else new_fitness
    
    if len(all_solutions) == 0:
        return archive
    
    # Map to grid cells
    normalized = normalize(all_solutions)
    normalized = np.clip(normalized, 0.0, 0.9999)
    grid_indices = (normalized * n_cells).astype(int)
    
    # Build grid: cell_key -> list of (fitness, solution_idx)
    grid = {}
    for idx in range(len(all_solutions)):
        key = tuple(grid_indices[idx])
        if key not in grid:
            grid[key] = []
        grid[key].append((all_fitness[idx], idx))
    
    # Select best per cell (up to cell_capacity) and compute diversity bonus
    selected_indices = []
    cell_best_fits = {}
    
    for key, members in grid.items():
        # Sort by fitness descending (lower error = better = higher value when negated)
        members_sorted = sorted(members, key=lambda x: x[0])
        keep_count = min(len(members_sorted), cell_capacity)
        
        for rank, (fit, sol_idx) in enumerate(members_sorted[:keep_count]):
            selected_indices.append(sol_idx)
            cell_best_fits[sol_idx] = fit - rank * 1e-10  # Small diversity bonus for rank
    
    # If still too many, trim by fitness with diversity consideration
    if len(selected_indices) > self.archive_max:
        selected_fitness = np.array([cell_best_fits[i] for i in selected_indices])
        keep_top = np.argsort(selected_fitness)[-self.archive_max:]
        selected_indices = [selected_indices[i] for i in keep_top]
    
    return all_solutions[selected_indices]
```