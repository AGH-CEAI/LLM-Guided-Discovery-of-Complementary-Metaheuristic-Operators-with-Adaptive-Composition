import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    A DE variant with:
    - Ring topology for mutation parent selection
    - Directional memory that biases mutation toward successful directions
    - Adaptive F and Cr based on historical success rates
    - Composite mutation: directional + rand/1 blended adaptively
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # Control parameters
        self.F_base = 0.5
        self.Cr_base = 0.3
        self.F = self.F_base
        self.Cr = self.Cr_base
        
        # Directional memory
        self.direction_memory_size = 5
        self.successful_directions = []
        self.direction_decay = 0.95
        
        # Adaptation tracking
        self.adaptation_window = 10
        self.F_success_history = []
        self.Cr_success_history = []
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize using Latin Hypercube Sampling for better spread."""
        sampler = np.linspace(self.lower, self.upper, self.np)
        population = np.zeros((self.np, self.dim))
        
        for d in range(self.dim):
            indices = np.random.permutation(self.np)
            offsets = np.random.uniform(0, 1, self.np)
            population[:, d] = sampler[indices] + offsets * (sampler[1] - sampler[0])
        
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            fitness = np.resize(fitness, self.np)
            fitness[np.isnan(fitness)] = np.inf
        
        best_idx = np.argmin(fitness)
        return population, fitness, fitness[best_idx], population[best_idx].copy()
    
    def _clip_to_bounds(self, population):
        """Hard clip all candidates to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        """Create ring neighbor indices for each population member."""
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        """Compute fitness-weighted centroid for directional bias."""
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
        """Track successful mutation directions for adaptive biasing."""
        if not np.any(successful_mask):
            self.successful_directions = [
                d * self.direction_decay for d in self.successful_directions
            ]
            return
        
        for mutation_vec in successful_mutations[successful_mask]:
            self.successful_directions.append(mutation_vec.copy())
        
        if len(self.successful_directions) > self.direction_memory_size:
            self.successful_directions = self.successful_directions[-self.direction_memory_size:]
    
    def _compute_directional_bias(self, target_indices):
        """Compute directional bias from historical successful mutations."""
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using composite strategy:
        - Ring-based rand/1 component
        - Directional component from historical memory
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        directional_bias = self._compute_directional_bias(np.arange(np_pop))
        
        # Select ring-based parents
        r1 = ring_prev
        r2 = ring_next
        
        # Standard DE rand/1 mutation
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F with perturbation based on diversity
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base mutation: ring-based differential
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Directional component: bias toward successful regions
        directional_component = directional_bias
        
        # Blend based on adaptation success
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (centroid + directional_component)
        
        return trials, current_F
    
    def _crossover_batch(self, population, mutant_batch):
        """Binomial crossover with adaptive Cr."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Per-individual Cr with bounded distribution
        Cr_individual = np.clip(
            self.Cr + 0.1 * np.random.randn(np_pop),
            0.1, 0.9
        )
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, dim, size=np_pop)
        
        for i in range(np_pop):
            mask = np.random.rand(dim) < Cr_individual[i]
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutant_batch[i], population[i])
        
        return trials, Cr_individual
    
    def _evaluate_batch(self, batch, func):
        """Evaluate batch, handle budget exhaustion gracefully."""
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
            """Greedy selection with spectral diversity modulation."""
            improved_mask = trial_fitness < fitness
            improvement = fitness - trial_fitness

            # Compute population covariance eigenvalue spread
            centered = population - np.mean(population, axis=0)
            cov = np.cov(centered.T)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            # Eigenvalue spread: ratio of largest to smallest non-zero eigenvalue
            # High spread → anisotropic population → reduce selection pressure
            if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
                condition = eigvals[0] / eigvals[-1]
            else:
                condition = 1.0

            # Modulation: in ill-conditioned populations, accept smaller improvements
            # log(condition) maps: 1→0, 10→2.3, 100→4.6, 1000→6.9
            diversity_bonus = np.log1p(condition)

            # Scaled improvement relative to population fitness range
            fit_range = np.ptp(fitness)
            fit_range = max(fit_range, 1e-10)
            scaled_improvement = improvement / fit_range

            # Accept if improved OR if scaled improvement exceeds diversity threshold
            accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
            accept_mask = improved_mask | (scaled_improvement > accept_threshold)

            new_population = population.copy()
            new_fitness = fitness.copy()

            new_population[accept_mask] = trials[accept_mask]
            new_fitness[accept_mask] = trial_fitness[accept_mask]

            return new_population, new_fitness, accept_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using k-NN graph topology properties."""
        # Build k-NN graph over population (using current population from optimizer)
        k = max(3, min(7, self.np // 10))
        pop = self.population_buffer if hasattr(self, 'population_buffer') else None

        if pop is not None and len(pop) >= k + 1:
            # Compute pairwise distances
            dists = np.linalg.norm(pop[:, np.newaxis] - pop[np.newaxis, :], axis=2)
            np.fill_diagonal(dists, np.inf)

            # k-NN adjacency: each node connects to its k nearest neighbors
            knn_indices = np.argsort(dists, axis=1)[:, :k]
            row_idx = np.arange(len(pop)).repeat(k)
            col_idx = knn_indices.ravel()

            # Build sparse adjacency (undirected)
            n = len(pop)
            adj = np.zeros((n, n), dtype=bool)
            adj[row_idx, col_idx] = True
            adj = adj | adj.T

            # Component analysis: count connected components
            visited = np.zeros(n, dtype=bool)
            components = []
            for start in range(n):
                if not visited[start]:
                    comp = []
                    stack = [start]
                    while stack:
                        node = stack.pop()
                        if not visited[node]:
                            visited[node] = True
                            comp.append(node)
                            neighbors = np.where(adj[node])[0]
                            for nb in neighbors:
                                if not visited[nb]:
                                    stack.append(nb)
                    components.append(comp)

            component_sizes = np.array([len(c) for c in components])
            n_components = len(components)
            main_component_frac = np.max(component_sizes) / n if n_components > 0 else 1.0

            # Clustering coefficient: for each node, fraction of neighbor pairs that are connected
            clustering_sum = 0.0
            for i in range(n):
                neighbors = np.where(adj[i])[0]
                deg = len(neighbors)
                if deg >= 2:
                    subgraph = adj[np.ix_(neighbors, neighbors)]
                    edges = np.sum(subgraph) / 2
                    possible = deg * (deg - 1) / 2
                    clustering_sum += edges / possible
            avg_clustering = clustering_sum / n if n > 0 else 0.0

            # Average shortest path in main component (approximate via sampling for efficiency)
            main_comp_nodes = np.where(component_sizes == np.max(component_sizes))[0]
            if len(main_comp_nodes) > 1:
                sample_size = min(20, len(main_comp_nodes))
                sample_nodes = np.random.choice(main_comp_nodes, sample_size, replace=False)
                path_lengths = []
                for src in sample_nodes:
                    dists_src = np.full(n, np.inf)
                    dists_src[src] = 0
                    queue = [src]
                    head = 0
                    while head < len(queue):
                        node = queue[head]
                        head += 1
                        for nb in np.where(adj[node])[0]:
                            if dists_src[nb] == np.inf:
                                dists_src[nb] = dists_src[node] + 1
                                queue.append(nb)
                    path_lengths.extend([d for d in dists_src[main_comp_nodes] if d < np.inf and d > 0])
                avg_path = np.mean(path_lengths) if path_lengths else 1.0
            else:
                avg_path = 1.0

            # Graph-based regime detection
            graph_fragmented = n_components > n / 3 or main_component_frac < 0.6
            graph_well_connected = n_components <= 2 and main_component_frac > 0.85
            graph_isolated_nodes = np.sum(component_sizes == 1) > n * 0.1

        else:
            n_components = 1
            main_component_frac = 1.0
            avg_clustering = 0.5
            avg_path = 1.0
            graph_fragmented = False
            graph_well_connected = True
            graph_isolated_nodes = False

        # Initialize graph history on first call
        if not hasattr(self, 'graph_n_components_history'):
            self.graph_n_components_history = []
            self.graph_clustering_history = []
            self.graph_path_history = []

        # Track graph properties over time
        self.graph_n_components_history.append(n_components)
        self.graph_clustering_history.append(avg_clustering)
        self.graph_path_history.append(avg_path)
        if len(self.graph_n_components_history) > 10:
            self.graph_n_components_history.pop(0)
            self.graph_clustering_history.pop(0)
            self.graph_path_history.pop(0)

        # Compute trend in graph connectivity
        if len(self.graph_n_components_history) >= 3:
            recent_frac_change = (self.graph_n_components_history[-1] - 
                                  np.mean(self.graph_n_components_history[-3:])) / max(1, np.mean(self.graph_n_components_history[-3:]))
        else:
            recent_frac_change = 0.0

        # Compute graph-based adjustment factor
        if graph_fragmented:
            # Fragmented graph: need more exploration
            component_entropy = -np.sum((component_sizes / n) * np.log(component_sizes / n + 1e-10))
            max_entropy = np.log(n_components + 1e-10)
            norm_entropy = component_entropy / (max_entropy + 1e-10)
            F_adjustment = 1.0 + 0.25 * norm_entropy + 0.1 * recent_frac_change
            Cr_adjustment = 0.85
            self.stagnation_counter = getattr(self, 'stagnation_counter', 0) + 1
        elif graph_isolated_nodes:
            # Many isolated nodes: need diversity injection
            F_adjustment = 1.15
            Cr_adjustment = 0.80
            self.stagnation_counter = getattr(self, 'stagnation_counter', 0) + 1
        elif graph_well_connected:
            # Well-connected graph: can exploit more
            F_adjustment = 0.90 - 0.1 * avg_clustering
            Cr_adjustment = 1.05 + 0.05 * (1.0 - avg_clustering)
            self.stagnation_counter = max(0, getattr(self, 'stagnation_counter', 0) - 1)
        else:
            # Moderate connectivity: balanced adjustment
            delta_comp = (np.mean(self.graph_n_components_history[-3:]) - 
                          np.mean(self.graph_n_components_history[:3])) if len(self.graph_n_components_history) >= 6 else 0.0
            F_adjustment = 1.0 + 0.05 * delta_comp - 0.05 * recent_frac_change
            Cr_adjustment = 1.0 + 0.02 * (0.5 - avg_clustering)

        # Apply adjustments with bounds
        F_adjustment = np.clip(F_adjustment, 0.7, 1.4)
        Cr_adjustment = np.clip(Cr_adjustment, 0.8, 1.2)

        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
        """Reinitialize portion of population if stagnation detected."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
        
        diversity = self._compute_diversity(population)
        
        if (self.stagnation_counter > self.stagnation_threshold or 
            diversity < self.min_diversity_threshold):
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Compute successful mutations for direction memory
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Adapt control parameters
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Check for stagnation and restart if needed
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            # Rebuild topology after potential restart
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
