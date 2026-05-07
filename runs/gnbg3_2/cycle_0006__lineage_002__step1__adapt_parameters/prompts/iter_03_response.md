**Idea: Entropy-Driven Distribution Adaptation**
One-line description: Fit the population as a Gaussian Mixture, compute entropy of the fitness distribution, use KL divergence from uniform to detect collapse, and drive F/Cr adaptation from distributional information rather than temporal success rates.

```python
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
```