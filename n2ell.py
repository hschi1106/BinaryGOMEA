import numpy as np
import subprocess, sys
from functools import lru_cache
import concurrent.futures as cf
import os
import pandas as pd
import random

TRYS = 50      # total runs to test
TRIALS = 10    # continuous success times to count as success
START_POP = 5
MAX_POP = 1000000
MAX_BS_STEPS = None

def run_gomea_once(ell: int, pop: int, vtr: int, timeout: int | None = None) -> tuple[int, str, str]:
    """run GOMEA once, return (returncode, stdout, stderr); do not raise exceptions."""
    cmd = [
        "./GOMEA",
        "--L", str(ell),
        "--problem", "7_6_5",
        "--populationSize", str(pop),
        "--time", "10000",
        "--FOS", "1",
        "--vtr", str(vtr),
        "--seed", str(random.randint(1, 1000000)),
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as e:
        print(f"[TIMEOUT] ELL={ell}, POP={pop}: {e}", file=sys.stderr, flush=True)
        return 124, "", f"TIMEOUT: {e}"

def _hit_vtr(stdout: str, stderr: str) -> bool:
    text = (stdout + stderr)
    return "VTR HIT!" in text

@lru_cache(maxsize=None)
def check_success(ell: int, pop: int, vtr: int) -> bool:
    """test whether a certain pop size continuously succeeds TRIALS times in TRYS times (i.e., hits 'VTR HIT!')."""
    success = 0
    for i in range(TRYS):
        rc, out, err = run_gomea_once(ell, pop, vtr, timeout=None)
        ok = _hit_vtr(out, err)
        if not ok:
            if rc != 0:
                print(f"[WARN] ELL={ell} POP={pop} run#{i+1} rc={rc}", file=sys.stderr, flush=True)
            return False
            success = 0
        success += 1
        if success >= TRIALS:
            break
        if (TRYS - i - 1 + success) < TRIALS:
            return False
    return True

def bracket_interval(ell: int, vtr: int, start_pop: int, max_pop: int | None):
    """
    find (lo, hi) such that lo fails, hi succeeds, and lo < hi.
    If start_pop succeeds: search downwards by halving to find failure;
    If start_pop fails: search upwards by doubling to find success.
    Possible outcomes:
      - If even pop=1 succeeds, return (0, 1)
      - If still fails at max_pop, return (last_fail, None)
    """
    if check_success(ell, start_pop, vtr):
        # half until fail
        hi = start_pop
        lo = 0
        probe = max(1, start_pop // 2)
        while probe >= 1:
            if check_success(ell, probe, vtr):
                hi = probe
                if probe == 1:
                    return 0, 1
                probe //= 2
            else:
                lo = probe
                break
        if lo == 0 and hi == start_pop:
            return 0, hi
        return lo, hi
    else:
        # plus 100 until success
        lo = start_pop
        hi = None
        probe = start_pop + 100
        while True:
            if max_pop is not None and probe > max_pop:
                return lo, None
            if check_success(ell, probe, vtr):
                hi = probe
                break
            lo = probe
            probe += 100
        return lo, hi

def binary_search_min_pop(ell: int, vtr: int, lo: int, hi: int, max_steps: int | None = None) -> int:
    """
    binary search to find the minimal pop in (lo, hi) that succeeds.
    guarantee: check_success(lo)==False (or lo==0 as failure), check_success(hi)==True.
    If max_steps is hit, return current hi (best-known success).
    """
    steps = 0
    while lo + 1 < hi:
        if max_steps is not None and steps >= max_steps:
            # print(f"[BS-LIMIT] ELL={ell}: reached max_steps={max_steps}, return upper bound {hi}", flush=True)
            return hi  
        mid = (lo + hi) // 2
        if check_success(ell, mid, vtr):
            hi = mid
        else:
            lo = mid
        steps += 1
    return hi

def solve_one_ell(ell: int):
    vtr = ell * 6 // 5
    lo, hi = bracket_interval(ell, vtr, START_POP, MAX_POP)
    if hi is None:
        return (ell, None, lo, None)
    min_ok = binary_search_min_pop(ell, vtr, lo, hi, max_steps=MAX_BS_STEPS)
    return (ell, min_ok, lo, hi)

if __name__ == "__main__":
    ells = np.arange(5, 20, 5, dtype=int)

    # MAX_WORKERS = min(len(ells), os.cpu_count() or 4)
    MAX_WORKERS = 12
    print(f"Using MAX_WORKERS={MAX_WORKERS}")

    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(solve_one_ell, int(ell)): int(ell) for ell in ells}
        for fut in cf.as_completed(futs):
            ell = futs[fut]
            try:
                ell, min_ok, lo, hi = fut.result()
                if min_ok is None:
                    print(f"[FAIL] ELL={ell}: last_fail={lo}, max_pop={MAX_POP}")
                else:
                    print(f"[RESULT] ELL={ell} min POP={min_ok} (lower={lo}, upper={hi})")
                results.append((ell, min_ok, lo, hi))
            except Exception as e:
                print(f"[ERROR] ELL={ell}: {e}")

    results.sort(key=lambda x: x[0])
    # Save to CSV
    df = pd.DataFrame(results, columns=["ELL", "Min_POP", "Last_Fail", "First_Success"])
    df.to_csv("n2ell_results.csv", index=False)
