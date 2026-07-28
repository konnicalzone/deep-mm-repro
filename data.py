import numpy as np
import itertools
from numba import jit

num_agents = 3


@jit(nopython=True)
def generate_preference_array(num_rows, num_agents):
    P = np.empty((num_rows, num_agents), dtype=np.float64)

    for row_idx in range(num_rows):
        permutation = np.random.permutation(num_agents)

        for agent_idx in range(num_agents):
            P[row_idx, agent_idx] = num_agents - 1 - permutation[agent_idx]+1

    return P


@jit(nopython=True)
def generate_permutation_array(batch_size, num_agents):
    N = batch_size * num_agents
    return generate_preference_array(N, num_agents)


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

    def generate_complete_misreports(self, P, Q, agent_idx, is_P, truncation=True):
        M = self.generate_complete_ranking(truncation=truncation)
        num_misreports = M.shape[-2]

        P_mis = np.tile(P[:, np.newaxis, :, :], [1, num_misreports, 1,1])
        Q_mis = np.tile(Q[:, np.newaxis, :, :], [1, num_misreports, 1,1])

        if is_P:
            P_mis[:, :, agent_idx, :] = M
        else:
            Q_mis[:, :, agent_idx, :] = M

        return P_mis, Q_mis

    def generate_batch(self, batch_size):
        N = batch_size * self.num_agents

        P = generate_preference_array(N, self.num_agents)
        Q = generate_preference_array(N, self.num_agents)

        P = self.add_unacceptable_options(P, truncation_probability=self.prob)
        Q = self.add_unacceptable_options(Q, truncation_probability=self.prob)

        P = P.reshape(batch_size, self.num_agents, self.num_agents)
        Q = Q.reshape(batch_size, self.num_agents, self.num_agents)

        if self.corr > 0:
            P_market = generate_preference_array(batch_size, self.num_agents)
            Q_market = generate_preference_array(batch_size, self.num_agents)

            P_market = self.add_unacceptable_options(P_market, truncation_probability=self.prob)
            Q_market = self.add_unacceptable_options(Q_market, truncation_probability=self.prob)

            P_market = P_market.reshape(batch_size, 1, self.num_agents)
            Q_market = Q_market.reshape(batch_size, 1, self.num_agents)

            P_corr = np.random.random((batch_size, self.num_agents, 1)) < self.corr
            Q_corr = np.random.random((batch_size, self.num_agents, 1)) < self.corr

            P = np.where(P_corr, P_market, P)
            Q = np.where(Q_corr, Q_market, Q)

        Q = np.transpose(Q, (0, 2, 1))
        return P, Q




if __name__ == "__main__":
    data = Data(num_agents=3, prob=0.5, corr=0.5)
    P, Q = data.generate_batch(batch_size=1)

    print("P shape:", P.shape)
    print(P)
    print("Q shape:", Q.shape)
    print(Q)
