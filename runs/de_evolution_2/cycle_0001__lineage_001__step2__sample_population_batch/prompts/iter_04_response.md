**Idea: Mirrored Sampling with Systematic Diversity**
Mirrored sampling generates paired offspring in opposite directions from the mean, ensuring symmetric exploration and preventing premature convergence on multimodal/deceptive landscapes where the current approach gets stuck at O(1) to O(100) error.
```python
def _sample_population_batch(self):
        """
        Sample a batch using mirrored sampling: half are independent samples,
        half are their systematic reflections. This guarantees symmetric
        exploration in all eigendirections and improves diversity on
        multimodal/deceptive tasks where standard sampling gets trapped.
        """
        half_size = self.pop_size // 2
        z = np.random.randn(half_size, self.dim)
        # Mirror: negate z for the second half
        z_mirrored = -z
        z_full = np.vstack([z, z_mirrored])
        # Transform: x = mean + sigma * B * D * z
        scaled = z_full * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        return population, z_full
```