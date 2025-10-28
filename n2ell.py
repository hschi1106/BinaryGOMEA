import numpy as np
import subprocess, sys
from functools import lru_cache
import concurrent.futures as cf
import os
import pandas as pd
import random
from tqdm.auto import tqdm   # 👈 新增

TRYS = 50      # total runs to test
TRIALS = 10    # continuous success times to count as success
START_POP = 5
MAX_POP = 1000000
MAX_BS_STEPS = None

def run_gomea_once(ell: int, pop: int, vtr: int, timeout: int | None = None) -> tuple[int, str, str]:
    cmd = [
        "./GOMEA", "--L", str(ell), "--problem", "7_6_5",
        "--populationSize", str(pop), "--time", "10000",
        "--FOS", "1", "--vtr", str(vtr),
        "--seed", str(random.randint(1, 1000000)),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as e:
        tqdm.write(f"[TIMEOUT] ELL={ell}, POP={pop}: {e}")  # 👈 用 tqdm.write
        return 124, "", f"TIMEOUT: {e}"

def _hit_vtr(stdout: str, stderr: str) -> bool:
    return "VTR HIT!" in (stdout + stderr)

@lru_cache(maxsize=None)
def check_success(ell: int, pop: int, vtr: int) -> bool:
    success = 0
    for i in range(TRYS):
        rc, out, err = run_gomea_once(ell, pop, vtr, timeout=None)
        ok = _hit_vtr(out, err)
        if not ok:
            if rc != 0:
                tqdm.write(f"[WARN] ELL={ell} POP={pop} run#{i+1} rc={rc}")  # 👈
            return False
        success += 1
        if success >= TRIALS:
            break
        if (TRYS - i - 1 + success) < TRIALS:
            return False
    return True

def bracket_interval(ell: int, vtr: int, start_pop: int, max_pop: int | None):
    if check_success(ell, start_pop, vtr):
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
    steps = 0
    while lo + 1 < hi:
        if max_steps is not None and steps >= max_steps:
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
    MAX_WORKERS = 12
    tqdm.write(f"Using MAX_WORKERS={MAX_WORKERS}")  # 👈 不破壞進度條

    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(solve_one_ell, int(ell)): int(ell) for ell in ells}
        with tqdm(total=len(ells), desc="ELLs done", unit="ell") as pbar:  # 👈 進度條
            for fut in cf.as_completed(futs):
                ell = futs[fut]
                try:
                    ell, min_ok, lo, hi = fut.result()
                    if min_ok is None:
                        tqdm.write(f"[FAIL]  ELL={ell}: last_fail={lo}, max_pop={MAX_POP}")
                    else:
                        tqdm.write(f"[RESULT] ELL={ell} min POP={min_ok} (lower={lo}, upper={hi})")
                    results.append((ell, min_ok, lo, hi))
                except Exception as e:
                    tqdm.write(f"[ERROR] ELL={ell}: {e}")
                pbar.update(1)  # 👈 每完成一個 ell 就+1

    results.sort(key=lambda x: x[0])
    pd.DataFrame(results, columns=["ELL", "Min_POP", "Last_Fail", "First_Success"]).to_csv("n2ell_results.csv", index=False)
