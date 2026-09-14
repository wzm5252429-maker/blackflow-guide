"""Paired empirical return targets from repeated offline future draws."""
from itertools import combinations
import numpy as np
import torch
from torch.nn import functional as F


def repeated_pairs(row, stable=False, minimum_gap=2.):
    indices=row['tested_indices']; values=np.asarray(row['replicated_returns'],dtype=float)
    if values.ndim!=2 or values.shape[1]!=len(indices) or values.shape[0]<2:
        raise ValueError('Need aligned repeated returns for every tested action')
    if len(indices)!=len(set(indices)) or not np.isfinite(values).all():
        raise ValueError('Duplicate action or invalid return')
    result=[]
    for a,b in combinations(range(len(indices)),2):
        deltas=values[:,a]-values[:,b]; mean=float(deltas.mean())
        if abs(mean)<minimum_gap: continue
        stderr=float(deltas.std(ddof=1)/np.sqrt(len(deltas)))
        signs=int(np.count_nonzero(deltas*np.sign(mean)>0))
        if stable and (signs < int(np.ceil(.75*len(deltas))) or abs(mean)<stderr):
            continue
        high,low=(a,b) if mean>0 else (b,a)
        result.append((indices[high],indices[low],min(abs(mean)/5.,4.),abs(mean),stderr))
    return result


def repeated_return_loss(logits,rows,stable=False):
    losses=[]
    for scores,row in zip(logits,rows):
        pairs=repeated_pairs(row,stable)
        if not pairs:
            losses.append(scores[:len(row['macros'])].sum()*0.)
            continue
        hi=torch.tensor([p[0] for p in pairs],device=scores.device)
        lo=torch.tensor([p[1] for p in pairs],device=scores.device)
        weights=scores.new_tensor([p[2] for p in pairs])
        losses.append((F.softplus(-(scores[hi]-scores[lo]))*weights).sum()/weights.sum())
    return torch.stack(losses)
