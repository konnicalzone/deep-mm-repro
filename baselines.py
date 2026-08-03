import itertools
import numpy as np
from numba import jit

def example_market():
    P = np.array([[
        [1/3, 1, 2/3],
        [2/3, 1, 1/3],
        [1, 1/3, 2/3],
    ]])

    Q = np.array([[
        [1, 1/3, 2/3],
        [2/3, 1, 1/3],
        [1/3, 2/3, 1],
    ]])

    menPreferences = np.argsort(-P, axis=2)
    womenPreferences = np.argsort(-np.swapaxes(Q, 1, 2), axis=2)
    order = [1,0,2,3,5,4]

    return P, Q, menPreferences, womenPreferences, order

@jit(nopython=True)
def numba_DA(P, Q, menPreferences, womenPreferences):

    num_instances, num_agents = P.shape[0], P.shape[1]
    R = np.zeros(P.shape)

    for inst in range(num_instances):

        unmarried_Men = list(range(num_agents))

        manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents

        nextManchoice = [0] * num_agents

        while unmarried_Men:

            he = unmarried_Men[0]

            if nextManchoice[he] == num_agents:
                manSpouse[he] = num_agents
                unmarried_Men.pop(0)
                continue

            she = menPreferences[inst, he, nextManchoice[he]]

            if P[inst, he, she] < 0:
                manSpouse[he] = num_agents
                unmarried_Men.pop(0)
                continue

            if womanSpouse[she] == -1:
                if Q[inst, he, she]> 0:
                    womanSpouse[she], manSpouse[he] = he, she
                    R[inst, he, she] = 1
                    unmarried_Men.pop(0)
            else:
                currentHusband = womanSpouse[she]
                if Q[inst, he, she] > Q[inst, currentHusband, she]:
                    womanSpouse[she], manSpouse[he] = he, she
                    R[inst, he, she] = 1
                    R[inst, currentHusband, she] = 0
                    unmarried_Men[0] = currentHusband

            nextManchoice[he] = nextManchoice[he] + 1

    return R

@jit(nopython=True)
def numba_SD(P,Q, menPreferences, womenPreferences, order):

    num_instances, num_agents = P.shape[0], P.shape[1]
    #Remember for extension that num_agents shape differing for market sides
    R = np.zeros(P.shape)

    for inst in range(num_instances):
        manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents
        for n in order:
            if n < num_agents:
                he = n
                if manSpouse[he] != -1:
                    continue

                for she in menPreferences[inst, he]:
                    if P[inst,he,she] < 0:
                        break

                    if womanSpouse[she] == -1:
                        manSpouse[he], womanSpouse[she] = she, he
                        R[inst, he, she] = 1
                        break
                if manSpouse[he] == -1:
                    manSpouse[he] = num_agents
            else:
                she = n - num_agents

                if womanSpouse[she] != -1:
                    continue

                for he in womenPreferences[inst, :, she]:

                    if Q[inst, he, she] < 0:
                        break

                    if manSpouse[he] == -1:
                        manSpouse[he], womanSpouse[she] = she, he
                        R[inst, he, she] = 1
                        break
                if womanSpouse[she] == -1:
                    womanSpouse[she] = num_agents
    return R

@jit(nopython=True)
def numba_TTC(P, Q, menPreferences, womenPreferences):

    num_instances, num_agents = P.shape[0], P.shape[1]
    R = np.zeros(P.shape)
    all_agents = 2*num_agents

    for inst in range(num_instances):
        matched = [0] * (all_agents)
        for rnd in range(all_agents):

            G  = np.arange(all_agents)
            for n in range(all_agents):
                if matched[n] == 1:
                    G[n] = -1
                    continue

                if n < num_agents:
                    he = n
                    for she in menPreferences[inst, he]:
                        if P[inst, he, she] < 0:
                            break
                        if matched[num_agents + she] == 0:
                            G[he] = num_agents + she
                            break

                else:
                    she = n - num_agents
                    for he in womenPreferences[inst, :, she]:
                        if Q[inst, he,she] < 0:
                            break
                        if matched[he] == 0:
                            G[num_agents + she] = he
                            break

            curr = -1
            for n in range(num_agents):
                if matched[n] == 0:
                    curr = n
                    break
            if curr == -1:
                break

            visited = [0] * all_agents
            while not visited[curr] == 1:
                visited[curr] = 1
                curr = G[curr]

            if curr == G[curr]:
                matched[curr] = 1
                continue

            if curr >= num_agents:
                curr = G[curr]

            visited = [0] * all_agents

            while not visited[curr] == 1:
                R[inst, curr, G[curr]-num_agents] = 1
                matched[curr], matched[G[curr]] = 1, 1
                visited[curr], visited[G[curr]] = 1, 1
                curr = G[G[curr]]

    return R


if __name__ == "__main__":
    P, Q, menPreferences, womenPreferences, order = example_market()
    R = numba_DA(P, Q, menPreferences, womenPreferences)
    R2 = numba_SD(P, Q, menPreferences, womenPreferences, order)
    R3 = numba_TTC(P, Q, menPreferences, womenPreferences)
    expected_R = np.array([[
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]])

    print("menPreferences:")
    print(menPreferences)
    print("womenPreferences:")
    print(womenPreferences)
    print("R(DA):")
    print(R.astype(int))
    print("R2(SD):")
    print(R2.astype(int))
    print("R3(TTC):")
    print(R3.astype(int))
    print("expected R:")
    print(expected_R)
    print("matches expected:", np.array_equal(R, expected_R))
