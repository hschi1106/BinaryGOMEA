import numpy as np
import subprocess, sys
from functools import lru_cache

TRIALS = 10    # continuous success times to count as success
START_POP = 10
MAX_POP = None  # no upper limit

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
    """test whether a certain pop size succeeds TRIALS times in a row (i.e., hits 'VTR HIT!')."""
    success = 0
    for i in range(TRIALS):
        rc, out, err = run_gomea_once(ell, pop, vtr, timeout=None)
        ok = _hit_vtr(out, err)
        if not ok:
            if rc != 0:
                print(f"[WARN] ELL={ell} POP={pop} run#{i+1} rc={rc}", file=sys.stderr, flush=True)
            return False
        success += 1
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
        # double until success
        lo = start_pop
        hi = None
        probe = start_pop * 2
        while True:
            if max_pop is not None and probe > max_pop:
                return lo, None
            if check_success(ell, probe, vtr):
                hi = probe
                break
            lo = probe
            probe *= 2
        return lo, hi

def binary_search_min_pop(ell: int, vtr: int, lo: int, hi: int) -> int:
    """
    binary search to find the minimal pop in (lo, hi) that succeeds.
    guarantee: check_success(lo)==False (or lo==0 as failure), check_success(hi)==True.
    """
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if check_success(ell, mid, vtr):
            hi = mid
        else:
            lo = mid
    return hi

if __name__ == "__main__":
    ells = np.arange(5, 401, 5, dtype=int)
    print(ells, flush=True)

    for ell in ells:
        vtr = ell * 6 // 5
        print(f"\n=== ELL={ell}, VTR={vtr} ===", flush=True)

        lo, hi = bracket_interval(ell, vtr, START_POP, MAX_POP)

        if hi is None:
            print(f"[FAIL] unable to solve under the limit: ELL={ell}, last_fail={lo}, max_pop={MAX_POP}", flush=True)
            continue

        min_ok = binary_search_min_pop(ell, vtr, lo, hi)
        print(f"[RESULT] ELL={ell} min POP={min_ok} (lower bound={lo}、upper bound={hi})", flush=True)
