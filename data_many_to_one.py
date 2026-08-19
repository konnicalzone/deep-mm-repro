import numpy as np
import itertools

from numba import jit



@jit(nopython=True)
def generate_preference_array(num_rows, num_items):
    P = np.empty((num_rows, num_items), dtype = np.float64)

    for row_idx in range(num_rows):
        P[row_idx] = np.random.permutation(num_items) +1

    return P

@jit(nopython=True)
def generate_permutation_array(batch_size, num_rankers, num_items):
    N = batch_size * num_rankers
    return generate_preference_array(N, num_items)

class Data(object):
    def __init__(self, n_students, n_schools, min_capacity, max_capacity, prob= 0.0, corr= 0.0):
        self.n_students = n_students
        self.n_schools = n_schools
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.prob = prob
        self.corr = corr
    #truncation solely added for students now, meaning assuming universities will accept anyone more than empty spot.
    def add_unacceptable_options(self, P, truncation_probability):
        return self._add_unacceptable_options(P, truncation_probability)/self.n_schools

    def sample_capacity(self, batch_size):
        return np.random.randint(
            self.min_capacity,
            self.max_capacity + 1,
            size=(batch_size, self.n_schools)
        )

    @staticmethod
    @jit(nopython=True)
    def _add_unacceptable_options(P, truncation_probability):
        N = P.shape[0]
        num_items = P.shape[1]
        P_trunc = P.copy()
        N_trunc = int(N * truncation_probability)

        if N_trunc > 0:
            sampled_rows = np.random.permutation(N)

            for sampled_idx in range(N_trunc):
                row_idx = sampled_rows[sampled_idx]
                threshold_col = np.random.randint(0, num_items)
                threshold_rank = P_trunc[row_idx, threshold_col]

                for col_idx in range(num_items):
                    P_trunc[row_idx, col_idx] = P_trunc[row_idx, col_idx] - threshold_rank

        return P_trunc

    def generate_complete_ranking(self, truncation = True):
        if not truncation:
            M = np.array(list(itertools.permutations(np.arange(self.n_schools)))) + 1
        else:
            M = np.array(list(itertools.permutations(np.arange(self.n_schools + 1))))
            M = (M - M[:,-1:])[:,:-1]
        return M/self.n_schools

    def generate_complete_misreports(self, P, Q, capacity, student_idx, truncation = True):
        M = self.generate_complete_ranking(truncation=truncation)
        num_misreports = M.shape[-2]

        P_mis = np.tile(P[:,np.newaxis,:,:], [1, num_misreports, 1, 1])
        Q_mis = np.tile(Q[:,np.newaxis,:,:], [1, num_misreports, 1, 1])
        capacity_mis = np.tile(capacity[:,np.newaxis,:], [1, num_misreports, 1])

        P_mis[:,:, student_idx,:] = M

        return P_mis, Q_mis, capacity_mis

    def _generate_preferences(self, num_rows, num_items, truncate=True):
        P = generate_preference_array(num_rows, num_items)
        if truncate:
            return self.add_unacceptable_options(P, truncation_probability=self.prob)
        else:
            return P/num_items

    def _blend_with_market(self, P, batch_size):
        num_rows = batch_size
        num_items = self.n_schools
        market = self._generate_preferences(num_rows, num_items, truncate=True).reshape(num_rows, 1, num_items)
        mask = np.random.random((batch_size, self.n_students, 1)) < self.corr
        return np.where(mask, market, P)

    def generate_batch(self, batch_size):
        N_students = batch_size * self.n_students
        N_schools = batch_size * self.n_schools

        P_shape = (batch_size, self.n_students, self.n_schools)
        Q_shape = (batch_size, self.n_schools, self.n_students)

        P = self._generate_preferences(N_students, self.n_schools, truncate=True).reshape(P_shape)
        Q = self._generate_preferences(N_schools, self.n_students, truncate = False).reshape(Q_shape)

        if self.corr > 0:
            P = self._blend_with_market(P, batch_size)

        capacity = self.sample_capacity(batch_size)

        return P, Q.transpose(0,2,1), capacity