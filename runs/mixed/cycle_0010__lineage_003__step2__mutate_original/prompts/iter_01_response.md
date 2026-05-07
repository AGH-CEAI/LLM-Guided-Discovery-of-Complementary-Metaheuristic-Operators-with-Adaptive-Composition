**Idea: DE/current-to-rand/1 Baseline**
Classic rotation-invariant mutation using convex combination with random base vector, fundamentally different from pbest-directed strategies.
```python
def _mutate_original(self, population, fitness):
    """DE/current-to-rand/1: Rotation-invariant mutation with random base"""
    NP = self.NP
    dim = self.dim
    F = self.F

    # Select three distinct random indices per individual
    r1 = np.random.randint(0, NP, size=NP)
    r2 = np.random.randint(0, NP, size=NP)
    r3 = np.random.randint(0, NP, size=NP)

    for i in range(NP):
        while r1[i] == i:
            r1[i] = (r1[i] + 1) % NP
        while r2[i] == i or r2[i] == r1[i]:
            r2[i] = (r2[i] + 1) % NP
        while r3[i] == i or r3[i] == r1[i] or r3[i] == r2[i]:
            r3[i] = (r3[i] + 1) % NP

    # K controls convex combination: 0.5 * (1 + rand) gives K in [0.5, 1.0]
    K = 0.5 * (1.0 + np.random.rand(NP))

    # DE/current-to-rand/1: v_i = x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3)
    donors = (population +
              K[:, np.newaxis] * (population[r1] - population) +
              F * (population[r2] - population[r3]))

    return np.clip(donors, self.lower, self.upper)
```