import numpy as np
import itertools
from numba import jit

num_agents = 3

@jit(nopython=True)
def generate_permutation_array(batch_size, num_agents):
    N = batch_size * num_agents
    P = np.empty((N, num_agents), dtype=np.float64)

    for row_idx in range(N):
        permutation = np.random.permutation(num_agents)

        for agent_idx in range(num_agents):
            P[row_idx, agent_idx] = num_agents - 1 - permutation[agent_idx]+1

    return P


class Data(object):
    def __init__(self, num_agents, prob=0.0, corr=0.0):
        self.num_agents = num_agents
        self.prob = prob
        self.corr = corr

    def add_unacceptable_options(self, P, truncation_probability):
        P_trunc = self._add_unacceptable_options(P, truncation_probability)
        return P_trunc / self.num_agents

    @staticmethod
    @jit(nopython=True)
    def _add_unacceptable_options(P, truncation_probability):
        N = P.shape[0]
        num_agents = P.shape[1]
        P_trunc = P.copy()
        N_trunc = int(N * truncation_probability)

        if N_trunc > 0:
            sampled_rows = np.random.permutation(N)

            for sampled_idx in range(N_trunc):
                row_idx = sampled_rows[sampled_idx]
                threshold_col = np.random.randint(0, num_agents)
                threshold_rank = P_trunc[row_idx, threshold_col]

                for col_idx in range(num_agents):
                    P_trunc[row_idx, col_idx] = P_trunc[row_idx, col_idx] - threshold_rank

        return P_trunc
    # Next part solely for reproduction purposes, as planned to use Gradient Descent for misreport in editorial.
    def generate_complete_ranking(self, truncation=True):
        if not truncation:
            M = np.array(list(itertools.permutations(np.arange(self.num_agents)))) + 1
        else:
            M = np.array(list(itertools.permutations(np.arange(self.num_agents + 1))))
            M = (M - M[:, -1:])[:, :-1]
        return M / self.num_agents



if __name__ == "__main__":
    data = Data(num_agents=num_agents)
    P = generate_permutation_array(batch_size=3, num_agents=data.num_agents)
    P_trunc = data.add_unacceptable_options(P, truncation_probability=0.8)
    print(P_trunc)
