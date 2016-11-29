# IRL_with_Constraints

We chose to start by implementing the general algorithms given by Ng et al. in: http://ai.stanford.edu/~ang/papers/icml00-irl.pdf

Following Paul Weng's advice, we will be looking at two structural properties:
 - known order over unknown reward values
 - R(s, a) is unimodal in action a (assuming that there is an order over actions)

## Implementation Note:

All arrays (policies, vectors, dynamics etc) are implemented using NumPy arrays.