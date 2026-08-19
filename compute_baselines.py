import numpy as np
from numba.cuda.libdevicedecl import args

import baselines
from data import Data
from utils import init_logger

class Args:
    num_agents: int = 4

    prob: float = 0.2

    corr: float = 0

    batch_size: int = 256

    num_tst_samples = 20480

    num_tst_batches = num_tst_samples // batch_size

    seed: int = 42

def STABILITY_VIOLATION_BATCH(P,Q,R):
    WP = np.maximum(P[:,:, np.newaxis, :] - P[:,:,:,np.newaxis], 0 )
    WQ = np.maximum(Q[:,:,np.newaxis,:] - Q[:,np.newaxis,:,:], 0 )

    T = (1 - np.sum(R, axis = 1, keepdims = True))
    S = (1 - np.sum(R, axis= 2, keepdims = True))

    RGT_1 = np.einsum('bjc,bijc->bic', R, WQ) + T * np.maximum(Q, 0)
    RGT_2 = np.einsum('bia,biac->bic', R, WP) + S * np.maximum(P, 0)

    REGRET = RGT_1 * RGT_2

    return REGRET.sum(-1).mean()


def IR_VIOLATION_BATCH(P, Q, R):
    IR_1 = R * np.maximum(-Q, 0)
    IR_2 = R * np.maximum(-P, 0)
    IR = IR_1 + IR_2
    return IR.sum(-1).mean()

def WELFARE_BATCH(P, Q, R):
    VAL_WF = ((P*R).sum(-1).mean() + (Q*R).sum(-2).mean())
    return VAL_WF

def IC_FOSD_VIOLATION_BATCH(P, Q, R, mechanism):
    batch_size = P.shape[0]
    num_agents = P.shape[1]

    IC_viol_P = np.zeros(num_agents)
    IC_viol_Q = np.zeros(num_agents)

    for agent_idx in range(num_agents):
        P_mis, Q_mis = G.generate_all_misreports(P,Q,agent_idx = agent_idx, is_P = True, include_truncation= True)
        R_mis = mechanism(P_mis.reshape(-1, num_agents, num_agents),
                          Q_mis.reshape(-1, num_agents, num_agents))
        R_mis = R_mis.reshape(batch_size, -1, num_agents, num_agents)

        R_diff = (R_mis[:,:, agent_idx, :] - R[:, None, agent_idx, :]) * (P[:, None, agent_idx, :] > 0)
        IDX = np.argsort(-P[:, agent_idx, :])
        IDX = np.tile(IDX[:, None, :], (1, R_mis_shape[1], 1))

        FOSD_viol = np.cumsum(np.take_along_axis(R_diff, IDX, axis= -1),-1)
        IC_viol_P[agent_idx] = np.maximum(FOSD_viol,0).max(-1).max(-1).mean(-1)

        P_mis, Q_mis = G.generate_all_misreports(P, Q, agent_idx=agent_idx, is_P=False, include_truncation=True)
        R_mis = mechanism(P_mis.reshape(-1, num_agents, num_agents),
                          Q_mis.reshape(-1, num_agents, num_agents))
        R_mis = R_mis.reshape(batch_size, -1, num_agents, num_agents)

        R_diff = (R_mis[:, :, :, agent_idx] - R[:, None, :, agent_idx]) * (Q[:, None, :, agent_idx] > 0)
        IDX = np.argsort(-Q[:, :, agent_idx])
        IDX = np.tile(IDX[:, None, :], (1, R_mis.shape[1], 1))

        FOSD_viol = np.cumsum(np.take_along_axis(R_diff, IDX, axis=-1), -1)
        IC_viol_Q[agent_idx] = np.maximum(FOSD_viol, 0).max(-1).max(-1).mean(-1)

    IC_viol = (IC_viol_P.mean() + IC_viol_Q.mean()) *0.5
    return IC_viol

def compute_violations(mech):
    np.random.seed(args.seed)
    random.seed(args.seed)
    VAL_ST_LOSS = float()
    VAL_IC_LOSS = float()
    VAL_WF = float()

    for j in range(args.num_tst_batches):
        P, Q = G.generate_batch(args.batch_size)
        R = mech(P, Q)

        ST_LOSS = STABILITY_VIOLATION_BATCH(P, Q, R) + IR_VIOLATION_BATCH(P, Q, R)
        IC_LOSS = IC_FOSD_VIOLATION_BATCH(P, Q, R, mech)
        WF = WELFARE_BATCH(P, Q, R)

        VAL_ST_LOSS += ST_LOSS
        VAL_IC_LOSS += IC_LOSS
        VAL_WF += WF

    VAL_ST_LOSS /= args.num_tst_batches
    VAL_IC_LOSS /= args.num_tst_batches
    VAL_WF /= args.num_tst_batches
    return VAL_ST_LOSS, VAL_IC_LOSS, VAL_WF

