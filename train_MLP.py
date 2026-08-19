import os
import time
import logging
import argparse
import numpy as np
from typing import List

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_logger
from MLP import Net

from data_many_to_one import Data

device = "cuda"

class Args:
    n_students: int = 0

    n_schools: int = 0

    min_capacity: int = 0

    max_capacity: int = 0

    prob: float = 0.0

    corr: float = 0.00

    lambd: float = 0.00

    #Neural params

    net_arch: List[int] = [256,256,256,256,256]

    act_fn = nn.LeakyReLU

    #Optimization params

    batch_size: int = 256

    num_accums: int = 1

    learning_rate: float = 5e-3

    max_iteration: int = 1000

    print_iter: int = 50
    val_iter: int = 100

    save_iter: int = 100

    num_val_samples: int = 2560

    num_tst_samples: int = 20480

    seed: int =42
    resume: bool= False

def torch_var(x):
    return torch.Tensor(x).to(device)

def compute_st(r, p, q, capacity):
    n_students = r.size(1)
    wp = F.relu(p[:,:,None,:]- p[:,:,:,None])
    wq = F.relu(q[:,:,None,:]- q[:,None,:,:])
    t = (capacity[:,None,:] - torch.sum(r, dim=1, keepdim=True))
    s = (1- torch.sum(r, dim= 2, keepdim = True))

    rgt_1 = torch.einsum('bjc,bijc->bic', r, wq) + t * F.relu(q)
    rgt_2 = torch.einsum('bia,biac->bic', r, wp) + s * F.relu(p)

    regret = rgt_1 * rgt_2
    return regret.sum(-1).sum(-1).mean()/n_students

def compute_ir(r, p, q):
    n_students = r.size(1)
    ir_1 = r * F.relu(-q)
    ir_2 = r * F.relu(-p)
    ir = ir_1 + ir_2
    return ir.sum(-1).sum(-1).mean()/n_students


def compute_ic_single(r, p, q, P, Q, C, student_idx):
    n_students = r.size(1)
    n_schools = r.size(2)
    P_mis, Q_mis , C_mis = G.generate_complete_misreports(P, Q, C, student_idx)
    p_mis, q_mis , capacity_mis = torch_var(P_mis), torch_var(Q_mis), torch_var(C_mis)
    r_mis = model(p_mis.view(-1, n_students, n_schools), q_mis.view(-1, n_students, n_schools), capacity_mis.view(-1, n_schools))
    r_mis = r_mis.view(*P_mis.shape)

    r_diff = (r_mis[:, :, student_idx, :] - r[:, None, student_idx, :]) * (p[:, None, student_idx, :] > 0).to(p.dtype)
    _, idx = torch.sort(-p[:, student_idx, :])

    idx = idx[:, None, :].repeat(1, r_mis.size(1), 1)
    fosd_viol = torch.cumsum(torch.gather(r_diff, -1, idx), -1)
    IC_viol = F.relu(fosd_viol).max(-1)[0].max(-1)[0].mean(-1)
    return IC_viol


""" IC Violation """


def compute_ic(r, p, q, P, Q, C):
    n_students = r.size(1)
    IC_viol_P = torch.zeros(n_students).to(device)

    for student_idx in range(n_students):
        IC_viol_P[student_idx] = compute_ic_single(r, p, q, P, Q, C, student_idx)

    IC_viol = IC_viol_P.mean()

    return IC_viol


def evaluate(model, G, batch_size, num_samples):
    model.eval()
    num_batches = num_samples // batch_size
    with torch.no_grad():
        val_st_loss = 0.0
        val_ic_loss = 0.0
        for j in range(num_batches):
            P, Q , C = G.generate_batch(args.batch_size)
            p, q, capacity = torch_var(P), torch_var(Q), torch_var(C)
            r = model(p, q, capacity)
            st_loss = compute_st(r, p, q, capacity)
            ic_loss = compute_ic(r, p, q, P, Q, C)
            val_st_loss += st_loss.item() / num_batches
            val_ic_loss += ic_loss.item() / num_batches
    return val_st_loss, val_ic_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-nst', '--n_students', action='store',
                        dest='n_students', required=True, type=int,
                        help='n students')

    parser.add_argument('-nsc', '--n_schools', action='store',
                        dest='n_schools', required=True, type=int,
                        help='n schools')

    parser.add_argument('-minc', '--min_capacity', action='store',
                        dest='min_capacity', required=True, type=int,
                        help='min capacity')

    parser.add_argument('-maxc', '--max_capacity', action='store',
                        dest='max_capacity', required=True, type=int,
                        help='max capacity')

    parser.add_argument('-p', '--prob', action='store',
                        dest='prob', required=True, type=float,
                        help='Truncation Probability')

    parser.add_argument('-c', '--corr', action='store',
                        dest='corr', required=True, type=float,
                        help='Correlation Probability')

    parser.add_argument('-l', '--lambd', action='store',
                        dest='lambd', required=True, type=float,
                        help='Lambda')

    parser.add_argument('-r', '--resume', action='store',
                        dest='resume', default=0, type=int,
                        help='Resume Training')

    cmd_args = parser.parse_args()

    args = Args()
    args.n_students = cmd_args.n_students
    args.n_schools = cmd_args.n_schools
    args.min_capacity = cmd_args.min_capacity
    args.max_capacity = cmd_args.max_capacity
    args.prob = cmd_args.prob
    args.corr = cmd_args.corr
    args.lambd = cmd_args.lambd
    args.resume = (cmd_args.resume == 1)

    """ Loggers """
    root_dir = os.path.join("experiments",
                            f"students_{args.n_students}",
                            f"schools_{args.n_schools}",
                            f"capacity_{args.min_capacity}_{args.max_capacity}",
                            f"corr_{args.corr:.2f}",
                            "MLP")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    log_fname = os.path.join(root_dir, "LOG_lambd_%.4f" % (args.lambd))
    if args.resume:
        logger = init_logger(log_fname, filemode='a')
    else:
        logger = init_logger(log_fname)

    model_path = os.path.join(root_dir, "CHECKPOINT_lambd_%.4f" % (args.lambd))

    """ Seed for reproducibility """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    G = Data(args.n_students, args.n_schools, args.min_capacity, args.max_capacity, args.prob, args.corr)
    model = Net(args.net_arch, args.act_fn, args.n_students, args.n_schools).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10000, 25000], gamma=0.5)

    iteration = 0

    if args.resume:
        checkpoint = torch.load(model_path)
        iteration = checkpoint['iteration']
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("*** Resuming Training ***")

    # Trainer
    tic = time.time()
    while iteration < args.max_iteration:

        # Reset opt
        opt.zero_grad()
        model.train()

        # Inference
        for _ in range(args.num_accums):
            P, Q , C= G.generate_batch(args.batch_size)
            p, q , capacity = torch_var(P), torch_var(Q), torch_var(C)
            r = model(p, q, capacity)

            # Compute loss
            st_loss = compute_st(r, p, q, capacity)

            if args.lambd < 1.0:
                ic_loss = compute_ic(r, p, q, P, Q, C)
            else:
                ic_loss = torch.tensor(0.0, device=device)

            total_loss = (st_loss * args.lambd + ic_loss * (1 - args.lambd)) / args.num_accums
            total_loss.backward()

        opt.step()
        scheduler.step()
        t_elapsed = time.time() - tic

        iteration += 1

        # Validation
        if iteration % args.print_iter == 0 or iteration == args.max_iteration:
            logger.info(
                "[iter]: %d, [t]: %f, [stv]: %f, [rgt]: %f" % (iteration, t_elapsed, st_loss.item(), ic_loss.item()))

        if iteration % args.save_iter == 0 or iteration == args.max_iteration:
            checkpoint = dict()
            checkpoint['iteration'] = iteration
            checkpoint['model'] = model.state_dict()
            checkpoint['opt'] = opt.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, model_path)

        if iteration % args.val_iter == 0 or iteration == args.max_iteration:
            num_samples = args.num_tst_samples if iteration == args.max_iteration else args.num_val_samples
            val_st_loss, val_ic_loss = evaluate(model, G, args.batch_size, num_samples)
            logger.info("\t[TEST]: %d, [stv]: %f, [rgt]: %f" % (iteration, val_st_loss, val_ic_loss))

