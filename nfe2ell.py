import sys, re, secrets, subprocess
import numpy as np
import pandas as pd
import concurrent.futures as cf
from functools import lru_cache
from decimal import Decimal, InvalidOperation
from tqdm import tqdm

# =======================
# 可調參數
# =======================
TRYS = 50            # 最多嘗試次數
TRIALS = 10          # 需要連續成功的次數
START_POP = 5000
MAX_POP = 1_000_000  # 上限保險桿
BRACKET_STEP = 1000   # 起始往上探測步長（線性增加）
MAX_BS_STEPS = None  # 二分上限步數；None 表示不限
MAX_WORKERS = 12     # 外層跨 ell 併發
# =======================

def run_gomea_once(ell: int, pop: int, vtr: int, timeout: int | None = None) -> tuple[int, str, str]:
    """跑一次 GOMEA，回傳 (returncode, stdout, stderr)；不丟例外。"""
    cmd = [
        "./GOMEA",
        "--L", str(ell),
        "--problem", "7_6_5",
        "--populationSize", str(pop),
        "--time", "10000",
        "--FOS", "1",
        "--vtr", str(vtr),
        "--seed", str(secrets.randbits(64)),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired as e:
        print(f"[TIMEOUT] ELL={ell}, POP={pop}: {e}", file=sys.stderr, flush=True)
        return -9, "", f"TIMEOUT: {e}"  # -9 = SIGKILL（僅作為標示）

def _hit_vtr(stdout: str, stderr: str) -> bool:
    return "VTR HIT!" in (stdout + stderr)

def _nfe_from_out_token(out: str) -> int:
    """NFE 一定在 out.split()[4]；支援 1,234,567 / 1.23e+06。抓不到回 0。"""
    try:
        token = out.split()[4]
    except Exception:
        return 0
    s = token.replace(",", "")
    try:
        return int(Decimal(s))
    except (InvalidOperation, ValueError):
        try:
            return int(float(s))
        except Exception:
            return 0

@lru_cache(maxsize=4096)
def check_success(ell: int, pop: int, vtr: int) -> tuple[bool, int]:
    """
    在 TRYS 次嘗試內，是否出現**連續** TRIALS 次的 'VTR HIT!'。
    回傳 (is_success, avg_nfe_of_that_streak)。失敗時 avg_nfe = 0。
    使用 lru_cache 以避免重複測同一個 (ell, pop, vtr)。
    """
    consecutive = 0
    nfe_list: list[int] = []
    for i in range(TRYS):
        rc, out, err = run_gomea_once(ell, pop, vtr, timeout=None)
        ok = _hit_vtr(out, err)
        if ok:
            consecutive += 1
            nfe_list.append(_nfe_from_out_token(out))
            if consecutive >= TRIALS:
                # 成功：回連續段的平均 NFE（抓不到就 0）
                return True, (int(np.mean(nfe_list)) if nfe_list else 0)
        else:
            # 不中斷地繼續嘗試，但連續計數歸零
            consecutive = 0
            nfe_list.clear()
            # 若 rc 非 0，僅警示；仍可能是隨機失敗，因此不中止、繼續找連續成功段
            if rc != 0:
                print(f"[WARN] ELL={ell} POP={pop} try#{i+1} rc={rc}", file=sys.stderr, flush=True)

        # 早停判斷：剩餘嘗試次數 + 目前連續成功，已不可能達到 TRIALS
        remaining = TRYS - (i + 1)
        if remaining + consecutive < TRIALS:
            break

    return False, 0

def bracket_interval(ell: int, vtr: int, start_pop: int, max_pop: int | None) -> tuple[int, int | None, int | None]:
    """
    尋找 (lo, hi)：lo 失敗、hi 成功（連續 TRIALS 成功），且 lo < hi。
    回傳 (lo, hi, hi_nfe)。若找不到成功 hi，回 (last_fail, None, None)。
    策略：
      - 若 start_pop 成功：往下對半找直到遇到失敗或到 1。
      - 若 start_pop 失敗：以 BRACKET_STEP 線性往上找成功（避免一次倍增太激進）。
    """
    ok, hi_nfe = check_success(ell, start_pop, vtr)
    if ok:
        hi = start_pop
        lo = 0
        probe = max(1, start_pop // 2)
        while probe >= 1:
            ok2, nfe2 = check_success(ell, probe, vtr)
            if ok2:
                hi = probe
                hi_nfe = nfe2
                if probe == 1:
                    return 0, 1, hi_nfe
                probe //= 2
            else:
                lo = probe
                break
        if lo == 0 and hi == start_pop:
            return 0, hi, hi_nfe
        return lo, hi, hi_nfe
    else:
        lo = start_pop
        step = BRACKET_STEP
        probe = start_pop + step
        while True:
            if max_pop is not None and probe > max_pop:
                return lo, None, None
            ok2, nfe2 = check_success(ell, probe, vtr)
            if ok2:
                return lo, probe, nfe2
            lo = probe
            probe += step
        # 不會到這裡

def binary_search_min_pop(ell: int, vtr: int, lo: int, hi: int, hi_nfe: int | None,
                          max_steps: int | None = None) -> tuple[int, int | None]:
    """
    在 (lo, hi) 上做二分搜尋，找**最小**可行的 pop；返回 (min_ok, min_nfe)。
    - 需要保證：check_success(lo)==False（或 lo==0 視為失敗）、check_success(hi)==True。
    - 會傳遞並更新「成功側」的 NFE（hi_nfe），最後回最小成功 pop 時的 NFE。
    - 若步數超過 max_steps，回目前 hi 與其 NFE（保守）。
    """
    steps = 0
    best_hi = hi
    best_nfe = hi_nfe
    while lo + 1 < hi:
        if max_steps is not None and steps >= max_steps:
            return best_hi, best_nfe
        mid = (lo + hi) // 2
        ok, nfe = check_success(ell, mid, vtr)
        if ok:
            hi = mid
            best_hi = mid
            best_nfe = nfe
        else:
            lo = mid
        steps += 1
    return best_hi, best_nfe

def solve_one_ell(ell: int) -> tuple[int, int | None, int, int | None, int | None]:
    """對單一 ell：先夾區間，再二分；固定回傳 5 個欄位。"""
    vtr = ell * 6 // 5
    lo, hi, hi_nfe = bracket_interval(ell, vtr, START_POP, MAX_POP)
    if hi is None:
        return (ell, None, lo, None, None)
    min_ok, min_nfe = binary_search_min_pop(ell, vtr, lo, hi, hi_nfe, max_steps=MAX_BS_STEPS)
    return (ell, min_ok, lo, hi, min_nfe)

if __name__ == "__main__":
    ells = np.arange(255, 305, 5, dtype=int)

    workers = MAX_WORKERS
    print(f"Using MAX_WORKERS={workers}")

    results: list[tuple[int, int | None, int, int | None, int | None]] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex, tqdm(
        total=len(ells), desc="Solving ELLs", dynamic_ncols=True
    ) as pbar:
        futs = {ex.submit(solve_one_ell, int(ell)): int(ell) for ell in ells}
        for fut in cf.as_completed(futs):
            ell = futs[fut]
            try:
                ell, min_ok, lo, hi, min_nfe = fut.result()
                if min_ok is None:
                    tqdm.write(f"[FAIL]  ELL={ell}: last_fail={lo}, max_pop={MAX_POP}")
                else:
                    tqdm.write(f"[RESULT] ELL={ell}, min POP={min_ok} (lower={lo}, upper={hi}), avg NFE={min_nfe}")
                results.append((ell, min_ok, lo, hi, min_nfe))
            except Exception as e:
                tqdm.write(f"[ERROR] ELL={ell}: {e}")
            finally:
                pbar.update(1)

    results.sort(key=lambda x: x[0])
    df = pd.DataFrame(results, columns=["ell", "min_n", "Last_Fail", "First_Success", "avg_nfe"])
    df.to_csv("nfe2ell_results.csv", index=False)
