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
            """Adapt F and Cr using information-theoretic distributional analysis of the population."""
            # Initialize distributional tracking state
            if not hasattr(self, 'entropy_history'):
                self.entropy_history = []
                self.prev_entropy = None
                self.distribution_shift_count = 0
                self.gmm_components_history = []
                self.kl_divergence_history = []

            # Compute entropy of the fitness distribution (Shannon entropy of fitness bins)
            fitness = self.population_fitness if hasattr(self, 'population_fitness') else None
            if fitness is None:
                fitness = np.array([0.5])

            n_bins = min(20, len(fitness) // 2 + 1)
            if n_bins < 3:
                n_bins = 3

            hist_counts, _ = np.histogram(fitness, bins=n_bins)
            hist_probs = hist_counts / (np.sum(hist_counts) + 1e-10)
            hist_probs = hist_probs[hist_probs > 0]
            current_entropy = -np.sum(hist_probs * np.log(hist_probs + 1e-10))
            max_entropy = np.log(n_bins)
            entropy_ratio = current_entropy / (max_entropy + 1e-10)

            # Fit Gaussian to population and measure anisotropy via condition number
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = np.maximum(eigvals, 1e-12)
            condition_number = eigvals[0] / eigvals[-1]
            anisotropy_factor = np.log1p(condition_number) / 10.0

            # Compute KL divergence from current population to uniform distribution over bounds
            # Using coordinate-wise uniform reference
            pop_range = self.upper - self.lower
            pop_std = np.std(self.population, axis=0)
            pop_std = np.maximum(pop_std, 1e-6)
            uniform_std_approx = pop_range / (2 * np.sqrt(12))
            kl_approx = np.sum(np.log(uniform_std_approx / pop_std + 1e-10))
            kl_approx = np.clip(kl_approx, -50, 50)
            kl_normalized = kl_approx / (self.dim + 1e-10)

            # Detect distribution collapse: low entropy + high KL (concentrated vs uniform)
            distribution_collapsed = (entropy_ratio < 0.4) and (kl_normalized > 1.5)
            distribution_uniform = (entropy_ratio > 0.75) and (kl_normalized < 0.5)
            distribution_shifting = False
            if self.prev_entropy is not None:
                entropy_change = abs(current_entropy - self.prev_entropy) / (abs(self.prev_entropy) + 1e-10)
                distribution_shifting = entropy_change > 0.3

            # Track history
            self.entropy_history.append(current_entropy)
            if len(self.entropy_history) > 20:
                self.entropy_history.pop(0)
            self.kl_divergence_history.append(kl_normalized)
            if len(self.kl_divergence_history) > 20:
                self.kl_divergence_history.pop(0)

            # Compute entropy trend over recent history
            if len(self.entropy_history) >= 5:
                recent_mean = np.mean(self.entropy_history[-5:])
                older_mean = np.mean(self.entropy_history[-10:-5]) if len(self.entropy_history) >= 10 else recent_mean
                entropy_trend = recent_mean - older_mean
            else:
                entropy_trend = 0.0

            # Information-theoretic F adaptation
            if distribution_collapsed:
                # Population concentrated: need maximum exploration
                F_adjustment = 0.65
                self.distribution_shift_count += 1
            elif distribution_shifting:
                # Distribution changing rapidly: moderate exploration
                F_adjustment = 0.85
                self.distribution_shift_count += 1
            elif distribution_uniform:
                # Population well-spread: can increase exploitation
                F_adjustment = 1.1
                self.distribution_shift_count = max(0, self.distribution_shift_count - 1)
            elif entropy_trend < -0.1:
                # Entropy declining: gradual diversity loss, counteract
                F_adjustment = 0.9
            else:
                # Neutral: correct based on KL divergence and anisotropy
                kl_bias = np.clip(kl_normalized * 0.1, -0.1, 0.1)
                anis_bias = np.clip(anisotropy_factor * 0.05, -0.05, 0.05)
                F_adjustment = 1.0 + kl_bias - anis_bias

            # Information-theoretic Cr adaptation
            if distribution_collapsed:
                # High diversity pressure: increase recombination to spread genes
                Cr_adjustment = 1.12
            elif entropy_ratio > 0.7:
                # High diversity: can afford more exploitation via crossover
                Cr_adjustment = 1.08
            elif entropy_ratio < 0.3:
                # Low diversity: reduce crossover to preserve existing good solutions
                Cr_adjustment = 0.88
            else:
                # Neutral regime
                Cr_adjustment = 1.0 + 0.05 * (entropy_ratio - 0.5)

            # Update state
            self.prev_entropy = current_entropy
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
