Looking at this problem, I need to analyze why all 24 tasks remain unsolved, especially the worst ones (Tasks 16-20 with errors ~10-80).

**Analysis of Failure Mode:**
- All tasks are UNSOLVED, meaning the target error of 1e-08 is not reached by any variant
- The worst tasks (16, 17, 20, 23, 19) have errors in the 10-80 range — these are likely highly multimodal, deceptive, or non-separable functions
- The current Thompson Sampling with Beta distributions has ALL operators performing poorly, so alpha/beta values converge to similar values → random selection
- When ALL operators fail, the adaptive mechanism provides no differentiation

**Key Insight:**
The current approach rewards based on improvement + diversity, but when stuck at high error levels (10^1 to 10^2), small improvements give similar rewards. The selector needs to **break the tie** and force diversity in operator usage when performance is uniformly poor.

**Strategy:**
Use a **variance-weighted softmax selection** that:
1. Computes mean reward per operator
2. Adds variance-based bonus to operators with uncertain/divergent performance
3. Uses softmax with temperature that adapts to convergence state
4. Forces operator cycling when all are stuck

**Idea: Variance-Weighted Adaptive Selection**
When all operators perform poorly (low absolute rewards), the variance-based exploration bonus ensures different operators get tried, preventing the algorithm from being locked into a suboptimal operator.

```python
def _select_operator_thompson(self):
    """Select operator using variance-weighted adaptive selection."""
    rewards = np.array([np.mean(r) if len(r) > 0 else 0.0 for r in self.operator_rewards.values()])
    reward_vars = np.array([np.var(r) if len(r) > 1 else 1.0 for r in self.operator_rewards.values()])
    
    reward_vars = np.clip(reward_vars, 1e-6, None)
    
    # Temperature inversely proportional to best reward — higher when stuck
    best_reward = np.max(rewards)
    temp = 5.0 if best_reward < 1.0 else max(0.5, 2.0 - best_reward)
    
    # Variance-weighted exploration bonus
    var_bonus = 0.5 * np.sqrt(reward_vars)
    
    # Compute adaptive scores
    scores = rewards + var_bonus
    
    # Softmax with temperature
    scores_shifted = scores - np.max(scores)
    exp_scores = np.exp(scores_shifted / temp)
    probs = exp_scores / np.sum(exp_scores)
    
    probs = np.clip(probs, 1e-10, 1.0)
    probs /= np.sum(probs)
    
    # Force exploration if selection is too concentrated
    if np.max(probs) > 0.8:
        probs = 0.7 * probs + 0.3 / self.num_operators
    
    self.current_operator = int(np.random.choice(self.num_operators, p=probs))
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```