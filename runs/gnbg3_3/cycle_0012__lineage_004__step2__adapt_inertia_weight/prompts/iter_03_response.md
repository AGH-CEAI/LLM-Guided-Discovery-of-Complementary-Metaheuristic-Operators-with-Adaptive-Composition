**Idea: Spectral Entropy Inertia Adaptation**
Adapt inertia weight using differential entropy of the population distribution (computed from covariance eigenvalues as a probability distribution). High entropy = scattered population → higher inertia for exploration. Low entropy = concentrated → lower inertia for exploitation. This is fundamentally different from condition-number-based spectral approaches by using information-theoretic reasoning on the eigenvalue distribution.

```python
def _adapt_inertia_weight(self):
        """Adapt inertia weight using spectral entropy of population distribution.
        
        Treats the normalized covariance eigenvalues as a probability distribution
        and computes its entropy. High entropy = diverse/scattered population →
        higher inertia (exploration). Low entropy = concentrated population →
        lower inertia (exploitation).
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Treat eigenvalues as a probability distribution (normalized)
            total = np.sum(eigenvalues)
            if total > 0:
                p = eigenvalues / total
            else:
                p = np.ones_like(eigenvalues) / len(eigenvalues)
            
            # Shannon entropy of eigenvalue distribution (bits)
            # H = -sum(p_i * log2(p_i))
            p = np.clip(p, 1e-10, 1.0)
            spectral_entropy = -np.sum(p * np.log2(p))
            
            # Normalize by maximum possible entropy (uniform distribution)
            max_entropy = np.log2(len(eigenvalues) + 1e-10)
            if max_entropy > 0:
                normalized_entropy = spectral_entropy / max_entropy
            else:
                normalized_entropy = 0.5
            normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)

            # Information-theoretic signal: map entropy to inertia
            # High entropy (scattered) → more exploration (higher w)
            # Low entropy (concentrated) → more exploitation (lower w)
            target_inertia = 0.4 + 0.55 * normalized_entropy

            # Time-based decay baseline (shrinks w over generations)
            decay = 0.729 - 0.15 * (self.generation / 1000)
            decay = np.clip(decay, 0.4, 0.95)

            # Blend entropy-driven target with time decay
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            # Exponential moving average for smooth transitions
            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```