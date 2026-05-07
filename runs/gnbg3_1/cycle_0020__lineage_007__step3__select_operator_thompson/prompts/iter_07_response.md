Looking at the task errors, the worst tasks (16, 17, 20, 23) are stuck at errors of 1e+1 to 5e+1, suggesting severe convergence failures. The current Thompson Sampling with Beta distributions may be too slow to adapt when stuck in local optima, and exploration may be insufficient.

I need a fundamentally different operator selection strategy: **adaptive softmax with stagnation-driven temperature**. This directly addresses the core issue - when stagnated, temperature increases to force exploration across operators; when improving, temperature decreases to exploit the best performer.

**Idea: Stagnation-Driven Softmax**
Use softmax probabilities with temperature dynamically scaled by stagnation level. When stuck, high temperature creates near-uniform exploration. When improving, low temperature focuses on the best operator. This directly addresses the failure mode of tasks stuck at ~1e+1.

```python
def _select_operator_thompson(self):
    """Select operator using adaptive softmax with stagnation-driven temperature."""
    # Compute mean reward for each operator
    qualities = np.zeros(self.num_operators)
    for op in range(self.num_operators):
        if len(self.operator_rewards[op]) > 0:
            qualities[op] = float(np.mean(self.operator_rewards[op]))
        else:
            qualities[op] = -5.0  # Unexplored operators start pessimistic
    
    # Adaptive temperature: high when stagnant (explore), low when improving (exploit)
    stagnation_ratio = float(self.stagnation_counter) / max(float(self.max_stagnation), 1.0)
    stagnation_ratio = np.clip(stagnation_ratio, 0.0, 1.0)
    temp = 0.01 + 4.99 * stagnation_ratio
    temp = float(np.clip(temp, 0.01, 5.0))
    
    # Softmax with numerical stability
    max_q = float(np.max(qualities))
    exp_q = np.zeros(self.num_operators)
    for op in range(self.num_operators):
        exp_q[op] = float(np.exp(np.clip(qualities[op] - max_q, -700.0, 700.0)))
    
    sum_exp = float(np.sum(exp_q))
    if sum_exp <= 0.0 or np.isnan(sum_exp) or np.isinf(sum_exp):
        probs = np.ones(self.num_operators) / self.num_operators
    else:
        probs = exp_q / sum_exp
    
    probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
    probs /= np.sum(probs)
    
    self.current_operator = int(np.random.choice(self.num_operators, p=probs))
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```