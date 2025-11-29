import json
import datetime
import os
import itertools
from collections import defaultdict
import math
import csv
from typing import Callable


log_path = '.'


def set_out_dir(out_dir: str):
    global log_path
    log_path = os.path.join(out_dir, "run_log.jsonl")


def log(event_type : str, payload : dict, mode : str = "a"):
    """
    Write a log entry as machine-readable JSON Lines.

    Parameters:
    ----------
    event_type: short string identifying the message
    payload: dict containing structured data
    """
    # Log file path (overwritten every run)
    global log_path

    # Open log file (overwrite mode)
    with open(log_path, mode) as log_file:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            "data": payload,
        }
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

# ---------------------------
# Core computation functions
# ---------------------------


def compute_paths_probs_values(N: int, lower_bound: Callable[[int], float], upper_bound: Callable[[int], float]):
    """
    Compute the combinatorial counts (paths_remaining), the conditioned up-probabilities (p_up),
    and the optimal value (opt_value, m(n,k)) on the full valid grid defined by bounds.

    The logic:
      - valid[n] = all integer k in [lower_bound(n), upper_bound(n)]
      - paths_remaining[(N,k)] = 1 for valid terminal k
      - paths_remaining[(n,k)] = paths_remaining[(n+1,k+1)] + paths_remaining[(n+1,k-1)]
      - p_up[(n,k)] = paths_remaining[(n+1,k+1)] / paths_remaining[(n,k)]  (or 0 if denom==0)
      - opt_value uses backward DP: opt_value[(n,k)] = max(k, p_up*m(n+1,k+1) + (1-p_up)*m(n+1,k-1))

    Returns:
       paths_remaining, p_up, opt_value, valid (dictionary of valid k lists per n)
    """
    paths_remaining = defaultdict(int)   # renamed from 'l'
    p_up = {}                            # renamed from 'p'
    opt_value = {}                       # renamed from 'm'
    valid = {}

    # Build the full grid of valid integer states at each time
    for n in range(N + 1):
        lo = int(math.ceil(lower_bound(n)))
        hi = int(math.floor(upper_bound(n)))
        valid[n] = list(range(lo, hi + 1))
        print(lo, hi)

    # Terminal condition for paths_remaining
    for k in valid[N]:
        paths_remaining[(N, k)] = 1

    # Backward recursion for paths_remaining on the entire valid grid
    for n in range(N - 1, -1, -1):
        for k in valid[n]:
            paths_remaining[(n, k)] = (
                paths_remaining.get((n + 1, k + 1), 0) +
                paths_remaining.get((n + 1, k - 1), 0)
            )

    # Compute p_up (conditioned up probability)
    for n in range(N):
        for k in valid[n]:
            denom = paths_remaining.get((n, k), 0)
            if denom == 0:
                p_up[(n, k)] = 0.0
            else:
                p_up[(n, k)] = paths_remaining.get((n + 1, k + 1), 0) / denom

    # Terminal condition for opt_value
    for k in valid[N]:
        opt_value[(N, k)] = k

    # Backward recursion for opt_value
    for n in range(N - 1, -1, -1):
        for k in valid[n]:
            pu = p_up.get((n, k), 0.0)
            up_val = opt_value.get((n + 1, k + 1), k + 1)
            down_val = opt_value.get((n + 1, k - 1), k - 1)
            cont = pu * up_val + (1 - pu) * down_val
            opt_value[(n, k)] = max(k, cont)

    return paths_remaining, p_up, opt_value, valid


def compute_reachability(N: int, valid: dict[int, list[int]]):
    """
    Forward reachability from X_0 = 0.
    Build boolean dict reachable[(n,k)] indicating whether node (n,k) can be visited
    by an unconstrained simple random walk starting at 0 and staying within bounds at each time.
    """
    reachable = {n: {k: 0 for k in valid[n]} for n in range(N + 1)}
    # start at X_0 = 0 only if 0 is inside valid[0]
    for k in valid[0]:
        reachable[0][k] = 0
    if 0 in valid[0]:
        reachable[0][0] = 1

    # forward propagation
    for n in range(1, N + 1):
        for k in valid[n]:
            val = 0
            if (k - 1) in valid[n - 1] and reachable[n - 1].get(k - 1, 0):
                val = 1
            if (k + 1) in valid[n - 1] and reachable[n - 1].get(k + 1, 0):
                val = 1
            reachable[n][k] = val

    return reachable


def compute_thresholds(N: int, opt_value: dict[tuple[int, int], float], valid: dict[int, list[int]], reachable: dict[int, dict[int, int]], tol: float = 1e-12):
    """
    Compute threshold x(n) = minimal k with opt_value(n,k) == k.
    Return two dictionaries:
       - threshold_all[n]   : min k among all valid[n] with opt_value==k (or None)
       - threshold_reach[n] : min k among reachable[n]==1 with opt_value==k (or None)
    """
    threshold_all = {}
    threshold_reach = {}
    for n in range(N + 1):
        eqs = [k for k in valid[n] if abs(
            opt_value.get((n, k), float('inf')) - k) < tol]
        threshold_all[n] = min(eqs) if eqs else None

        # if reachable is not None:
        eqs_r = [k for k in valid[n] if reachable[n].get(k, 0) and abs(
            opt_value.get((n, k), float('inf')) - k) < tol]
        threshold_reach[n] = min(eqs_r) if eqs_r else None
        # else:
        #    threshold_reach[n] = None
    return threshold_all, threshold_reach
