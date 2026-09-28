"""Training-only branch execution and return comparisons for public route actions.

The real simulator state is used to *execute* alternatives offline. Only the
ordinary public encoding and candidate descriptors enter the neural network.
One branch return is a conditional simulator outcome, not an exact expected Q.
"""
from itertools import combinations

import torch
from torch.nn import functional as F

from .domain import ActionKind
from .policy_constraints import allowed_action_ids


def execute_macro(sim, state, macro, constraints):
    """Execute exactly the selected joint action, including its promised move."""
    if macro.first_action not in allowed_action_ids(sim, state, constraints):
        raise ValueError('Branch first action is no longer scope legal')
    after = sim.transition(state, macro.first_action).next_state
    if macro.requires_equip:
        allowed = set(allowed_action_ids(sim, after, constraints))
        onward = next((a for a in sim.legal_actions(after)
                       if a.action_id in allowed and a.kind == ActionKind.MOVE
                       and a.target_node_id == macro.target
                       and a.equipment_instance_id == macro.gear), None)
        if onward is None or after.equipped_instance_id != macro.gear:
            raise ValueError('Branch joint action lost its legal destination')
        after = sim.transition(after, onward.action_id).next_state
    return after


def shortlist(macros, logits, teacher_positive, limit=4):
    """Pick public alternatives before observing any branch outcome.

    Include the source policy, teacher suggestion and immediate concept option;
    then fill with source-ranked alternatives. Equivalent items get one trial.
    """
    if limit < 3:
        raise ValueError('Need space for source, teacher and concept candidates')
    order = sorted(range(len(macros)), key=lambda i: (-float(logits[i]), i))
    teacher = next((i for i in order if i in teacher_positive), order[0])
    payoff = max(range(len(macros)), key=lambda i: (macros[i].facts[10] + macros[i].facts[11], float(logits[i]), -i))
    chosen = []; semantics = set()
    for i in [order[0], teacher, payoff, *order]:
        if macros[i].semantic not in semantics:
            chosen.append(i); semantics.add(macros[i].semantic)
        if len(chosen) == limit:
            break
    return chosen


def return_pairs(row, minimum_gap=2.):
    """Only compare executed branches; never label an untested action bad."""
    branches = row['branches']
    if any(b['error'] or not b['ledger_valid'] for b in branches):
        raise ValueError('Erroneous branch cannot supply a return label')
    pairs = []
    for a, b in combinations(branches, 2):
        # A step-limited trajectory receives zero completion-gated return.
        # Its inventory is retained separately in the audit, not called final.
        qa = a['final_relics'] if a['completed'] else 0.
        qb = b['final_relics'] if b['completed'] else 0.
        if abs(qa - qb) >= minimum_gap:
            high, low = (a, b) if qa > qb else (b, a)
            pairs.append((high['index'], low['index'], abs(qa - qb)))
    return pairs


def pairwise_return_loss(logits, rows, minimum_gap=2.):
    """Episode weighting is applied by the caller; normalize within each state."""
    losses = []
    for scores, row in zip(logits, rows):
        pairs = return_pairs(row, minimum_gap)
        if not pairs:
            losses.append(scores[:len(row['macros'])].sum() * 0.)
            continue
        hi = torch.tensor([p[0] for p in pairs], device=scores.device)
        lo = torch.tensor([p[1] for p in pairs], device=scores.device)
        weights = scores.new_tensor([min(p[2] / 5., 4.) for p in pairs])
        losses.append((F.softplus(-(scores[hi] - scores[lo])) * weights).sum() / weights.sum())
    return torch.stack(losses)
