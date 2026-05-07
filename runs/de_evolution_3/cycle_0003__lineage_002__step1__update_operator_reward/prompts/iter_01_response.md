**Idea: UCB (Upper Confidence Bound) Baseline**
Replace Thompson Sampling with UCB, a well-established multi-armed bandit algorithm with strong theoretical guarantees. UCB balances exploration and exploitation deterministically using confidence bounds, making it a robust baseline for adaptive operator selection.

```python
def _update_operator_reward(self, operator_idx, reward):
    """Update UCB parameters based on reward."""
    operator_idx = int(operator_idx)
    reward = float(np.clip(reward, 0.0, 1.0))

    # Initialize tracking if first call
    if not hasattr(self, 'op_total_reward') or self.op_total_reward is None:
        self.op_total_reward = np.zeros(self.n_operators)
        self.op_count = np.zeros(self.n_operators)
        self.op_total_plays = 0

    # Ensure arrays are correct size (handles restart scenarios)
    if len(self.op_total_reward) != self.n_operators:
        self.op_total_reward = np.zeros(self.n_operators)
        self.op_count = np.zeros(self.n_operators)
        self.op_total_plays = 0

    # Update cumulative statistics
    self.op_total_reward[operator_idx] += reward
    self.op_count[operator_idx] += 1
    self.op_total_plays += 1

    # UCB exploration parameter (standard c=sqrt(2))
    c = np.sqrt(2.0)

    # Compute UCB scores for all operators
    ucb_scores = np.zeros(self.n_operators)
    eps = 1e-20
    for i in range(self.n_operators):
        if self.op_count[i] > 0:
            avg_reward = self.op_total_reward[i] / self.op_count[i]
            exploration = c * np.sqrt(np.log(self.op_total_plays + 1) / (self.op_count[i] + eps))
            ucb_scores[i] = avg_reward + exploration
        else:
            # Unexplored operators get infinite score
            ucb_scores[i] = np.inf

    # Store best operator based on UCB (used by selector)
    self.current_operator = int(np.argmax(ucb_scores))

    # Periodic reset to prevent saturation (every 500 plays)
    if self.op_total_plays % 500 == 0:
        self.op_total_reward *= 0.5
        self.op_count *= 0.5
```