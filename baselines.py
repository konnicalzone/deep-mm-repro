import numpy as np
from numba import jit
import itertools

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
def numba_RSD(P,Q,menPreferences, womenPreferences, orders):

    num_instances, num_agents = P.shape[0], P.shape[1]
    R = np.zeros(P.shape)
    all_agents = 2*num_agents

    for inst in range(num_instances):
        for order in orders:
            manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents
            for n in order:
                if n < num_agents:
                    he = n
                    if manSpouse[he] != -1:
                        continue

                    for she in menPreferences[inst, he]:
                        if P[inst, he, she] < 0:
                            break

                        if womanSpouse[she] == -1:
                            manSpouse[he], womanSpouse[she] = she, he
                            R[inst, he, she] += 1
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
                            R[inst, he, she] += 1
                            break
                    if womanSpouse[she] == -1:
                        womanSpouse[she] = num_agents

    return R/orders.shape[0]




@jit(nopython=True)
def numba_one_RSD(P, Q, menPreferences, womenPreferences, orders):
    num_instances, num_agents = P.shape[0], P.shape[1]
    R = np.zeros(P.shape)

    for inst in range(num_instances):
        for order in orders:
            womanSpouse = [-1] * num_agents
            for he in order:

                for she in menPreferences[inst,he]:

                    if P[inst, he, she] < 0:
                        break
                    if womanSpouse[she] == -1:
                        womanSpouse[she] = he
                        R[inst, he, she] += 1
                        break
    return R/orders.shape[0]

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

@jit(nopython=True)
def numba_BSC(P, Q, menPreferences, womenPreferences):
    num_instances, num_agents = P.shape[0], P.shape[1]
    R = np.zeros(P.shape)

    for inst in range(num_instances):
        manSpouse, womanSpouse = [-1] * num_agents, [-1] * num_agents

        for t in range(num_agents):

            currWomanSpouse = [-1] * num_agents

            for he in range(num_agents):

                if manSpouse[he] != -1:
                    continue

                she = menPreferences[inst, he, t]

                if womanSpouse[she] != -1:
                    continue
                if P[inst, he, she] < 0:
                    manSpouse[he] = num_agents
                    continue

                if currWomanSpouse[she] == -1:
                    if Q[inst,he,she] > 0:
                        R[inst, he, she] = 1
                        currWomanSpouse[she] = he

                else:
                    if Q[inst, he, she] > Q[inst, currWomanSpouse[she], she]:
                        R[inst, he, she] = 1
                        R[inst, currWomanSpouse[she], she] = 0
                        currWomanSpouse[she] = he

            for she in range(num_agents):
                if currWomanSpouse[she] != -1:
                    womanSpouse[she] = currWomanSpouse[she]
                    manSpouse[currWomanSpouse[she]] = she
    return R


#Worker proposing calling fcts

def compute_RSD_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(2 * P.shape[1])))))
    return numba_RSD(P, Q, menPreferences, womenPreferences, orders)

def compute_one_RSD_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(P.shape[1])))))
    return numba_RSD(P, Q, menPreferences, womenPreferences, orders)

def compute_TTC_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_TTC(P, Q, menPreferences, womenPreferences)

def compute_DA_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_DA(P, Q, menPreferences, womenPreferences)

def compute_BSC_batch(P, Q):
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_BSC(P, Q, menPreferences, womenPreferences)

def compute_one_RSD_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    orders = np.array(list(itertools.permutations(list(range(P.shape[1])))))
    return numba_RSD(P, Q, menPreferences, womenPreferences, orders).transpose(0, 2, 1)

def compute_TTC_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_TTC(P, Q, menPreferences, womenPreferences).transpose(0, 2, 1)

def compute_DA_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_DA(P, Q, menPreferences, womenPreferences).transpose(0, 2, 1)

def compute_BSC_batch_switch(P, Q):
    P, Q = Q.transpose((0, 2, 1)), P.transpose((0, 2, 1))
    menPreferences = np.argsort(-P, axis = -1)
    womenPreferences = np.argsort(-Q, axis = -2)
    return numba_BSC(P, Q, menPreferences, womenPreferences).transpose(0, 2, 1)
