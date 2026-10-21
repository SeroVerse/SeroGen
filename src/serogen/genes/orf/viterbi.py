r"""Numba-accelerated 29-state Viterbi forward dynamic programming and backtracking."""

import math

import numpy as np
from numba import njit

# HMM States -----------------------------------------------------------------------------------------------------------
S = 0
E = 1
R = 2
Sr = 3
Er = 4
M1 = 5
M2 = 6
M3 = 7
M4 = 8
M5 = 9
M6 = 10
M1r = 11
M2r = 12
M3r = 13
M4r = 14
M5r = 15
M6r = 16
I1 = 17
I2 = 18
I3 = 19
I4 = 20
I5 = 21
I6 = 22
I1r = 23
I2r = 24
I3r = 25
I4r = 26
I5r = 27
I6r = 28
NUM_STATES = 29

# Global Transition Penalty Indices ------------------------------------------------------------------------------------
TR_MM = 0
TR_MI = 1
TR_MD = 2
TR_II = 3
TR_IM = 4
TR_DD = 5
TR_DM = 6
TR_GE = 7
TR_GG = 8
TR_ER = 9
TR_RS = 10
TR_RR = 11
TR_ES = 12
TR_ES1 = 13

# Mathematical Precomputed Constants -----------------------------------------------------------------------------------
LN_025 = math.log(0.25)
LN_095 = math.log(0.95)


@njit(fastmath=True, nogil=True, cache=True)
def _modify_border_dist(
    cell_val: float,
    values: np.ndarray,
    start_freq: float,
) -> float:
    h_kd = values[2] * math.exp(-1.0 * (start_freq - values[1]) ** 2 / (values[0] ** 2) / 2.0)
    r_kd = values[5] * math.exp(-1.0 * (start_freq - values[4]) ** 2 / (values[3] ** 2) / 2.0)
    denom = h_kd + r_kd
    ratio = 0.01 if denom == 0.0 else h_kd / denom
    if ratio < 0.01:
        ratio = 0.01
    elif ratio > 0.99:
        ratio = 0.99
    return cell_val - math.log(ratio)


@njit(fastmath=True, nogil=True, cache=True)
def _build_local_cg_bins(
    seq: np.ndarray,
    default_cg_bin: int,
    window: int,
) -> np.ndarray:
    r"""Compute per-position local GC bins (0..43) using an O(N) prefix-sum sliding window."""
    n = len(seq)
    bins = np.empty(n, dtype=np.int8)
    if window <= 0 or n < 200:
        bins[:] = default_cg_bin
        return bins
    prefix = np.zeros(n + 1, dtype=np.int32)
    acc = 0
    for i in range(n):
        c = seq[i]
        if c == 1 or c == 2:
            acc += 1
        prefix[i + 1] = acc
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + window)
        lo = max(0, hi - window)
        span = hi - lo
        cg_pct = ((prefix[hi] - prefix[lo]) * 100) // span
        b = cg_pct - 26
        if b < 0:
            b = 0
        elif b > 43:
            b = 43
        bins[i] = b
    return bins


@njit(fastmath=True, nogil=True, cache=True)
def _build_tri_array(seq: np.ndarray) -> np.ndarray:
    r"""Precompute per-position 64-state trinucleotide indices in a single linear pass."""
    n = len(seq)
    tri = np.zeros(n, dtype=np.uint8)
    for i in range(n - 2):
        n0 = seq[i]
        n1 = seq[i + 1]
        n2 = seq[i + 2]
        if n0 <= 3 and n1 <= 3 and n2 <= 3:
            tri[i] = (n0 << 4) | (n1 << 2) | n2
    return tri


@njit(inline="always", fastmath=True, nogil=True)
def _avoid_insertion_stop(
    seq: np.ndarray,
    stop_mask: np.ndarray,
    phase: int,
    temp_idx: int,
    t: int,
    min_temp_idx: int,
) -> bool:
    r"""Check whether transitioning from an insertion state into `phase` forms an in-frame stop codon."""
    seq_len = len(seq)
    if phase == 0 or phase == 2 or phase == 5:
        if t >= 2 and temp_idx > min_temp_idx:
            b0, b1, b2 = seq[temp_idx - 1], seq[temp_idx], seq[t]
            if b0 <= 3 and b1 <= 3 and b2 <= 3 and stop_mask[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                return True
    elif phase == 1 or phase == 4:
        if t >= 2 and t + 1 < seq_len:
            b0, b1, b2 = seq[temp_idx], seq[t], seq[t + 1]
            if b0 <= 3 and b1 <= 3 and b2 <= 3 and stop_mask[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                return True
    return False


@njit(fastmath=True, nogil=True, cache=True)
def _viterbi_forward(
    seq: np.ndarray,
    cg_bins: np.ndarray,
    whole_genome: bool,
    include_phage: bool,
    tr: np.ndarray,
    tr_mi: np.ndarray,
    tr_ii: np.ndarray,
    pi: np.ndarray,
    tr_rr: np.ndarray,
    e_match: np.ndarray,
    tr_start: np.ndarray,
    tr_stop: np.ndarray,
    dist_start: np.ndarray,
    dist_stop: np.ndarray,
    start_mask_64: np.ndarray,
    stop_mask_64: np.ndarray,
    rev_start_mask_64: np.ndarray,
    rev_stop_mask_64: np.ndarray,
    start_log_priors_64: np.ndarray,
    rev_start_log_priors_64: np.ndarray,
    stop_log_priors_64: np.ndarray,
    rev_stop_log_priors_64: np.ndarray,
) -> tuple[int, np.ndarray]:
    seq_len = len(seq)
    tri_arr = _build_tri_array(seq)
    alpha = np.zeros((4, NUM_STATES), dtype=np.float64)
    path = np.empty((seq_len, NUM_STATES), dtype=np.int8)
    path[0, :] = 0
    if seq_len > 1:
        path[1, :] = 0
    if seq_len > 2:
        path[2, :] = 0

    temp_i = np.zeros(6, dtype=np.int64)
    temp_i_1 = np.zeros(6, dtype=np.int64)

    for i in range(NUM_STATES):
        alpha[0, i] = -pi[i]

    if seq_len >= 3:
        tri0 = int(tri_arr[0])
        if stop_mask_64[tri0]:
            alpha[0, E] = np.inf
            alpha[1, E] = np.inf
            path[1, E] = E
            path[2, E] = E
            alpha[2, M6] = np.inf
            alpha[1, M5] = np.inf
            alpha[0, M4] = np.inf
            alpha[2, M3] = np.inf
            alpha[1, M2] = np.inf
            alpha[0, M1] = np.inf
            alpha[2, E] -= stop_log_priors_64[tri0]

        if rev_stop_mask_64[tri0]:
            alpha[0, Sr] = np.inf
            alpha[1, Sr] = np.inf
            alpha[2, Sr] = alpha[0, S]
            path[1, Sr] = Sr
            path[2, Sr] = Sr
            alpha[2, M3r] = np.inf
            alpha[2, M6r] = np.inf
            alpha[2, Sr] = rev_stop_log_priors_64[tri0]

    tr_mm = tr[TR_MM]
    tr_gg = tr[TR_GG]
    tr_im_ln025 = (tr[TR_IM] if tr[TR_IM] < 0.0 else -0.10536) + LN_025
    tr_ii_pen = tr[TR_II] if tr[TR_II] > -20.0 else -3.0
    tr_mi_pen = tr[TR_MI] if tr[TR_MI] > -20.0 else -14.5
    tr_md_eff = tr[TR_MD] if tr[TR_MD] > -20.0 else -14.8
    tr_dd_eff = tr[TR_DD] if tr[TR_DD] > -20.0 else -3.0
    tr_dm_eff = tr[TR_DM] if tr[TR_DM] < 0.0 else -0.10536
    tr_ge = tr[TR_GE]
    tr_er_ln095 = tr[TR_ER] + LN_095
    tr_rr_ln095 = tr[TR_RR] + LN_095
    tr_rs = tr[TR_RS]
    tr_es = tr[TR_ES]
    tr_es1 = tr[TR_ES1]
    stop_rt_pen = 4.5
    del2_pen = tr_md_eff + LN_025 + tr_dm_eff
    del3_pen = tr_md_eff + 2.0 * LN_025 + tr_dd_eff + tr_dm_eff

    num_noncoding = 0
    for t in range(1, seq_len):
        cg_bin = int(cg_bins[t])
        r_curr = t & 3
        r_prev = (t - 1) & 3
        r_next1 = (t + 1) & 3
        r_next2 = (t + 2) & 3
        if t + 2 < seq_len:
            for st_clear in range(NUM_STATES):
                alpha[r_next2, st_clear] = 0.0
        from_nt = seq[t - 1]
        if from_nt > 3:
            from_nt = 2
        if t > 1:
            from0 = seq[t - 2]
            if from0 > 3:
                from0 = 2
        else:
            from0 = 2

        to_nt = seq[t]
        if to_nt > 3:
            num_noncoding += 1
            to_nt = 2
        else:
            num_noncoding = 0

        from2 = from0 * 4 + from_nt
        em_f = e_match[0, cg_bin]
        em_r = e_match[1, cg_bin]

        # 1. Forward Match States M1..M6
        for phase in range(6):
            m_cur = M1 + phase
            if alpha[r_curr, m_cur] >= 1e300:
                continue
            em = em_f[phase, from2, to_nt]
            if (phase == 2 or phase == 5) and em < -5.2:
                em = -5.2
            p_prev1 = (phase + 5) % 6
            p_prev2 = (phase + 4) % 6
            p_prev3 = (phase + 3) % 6
            m_prev1 = M1 + p_prev1
            m_prev2 = M1 + p_prev2
            m_prev3 = M1 + p_prev3

            val_m = alpha[r_prev, m_prev1] - (tr_gg if phase == 0 else 0.0) - tr_mm - em
            p_m = m_prev1

            temp_d2 = alpha[r_prev, m_prev2] - em - del2_pen
            if temp_d2 < val_m:
                val_m = temp_d2
                p_m = m_prev2

            temp_d3 = alpha[r_prev, m_prev3] - em - del3_pen
            if temp_d3 < val_m:
                val_m = temp_d3
                p_m = m_prev3

            if phase == 0:
                temp_s = alpha[r_prev, S] - em
                if temp_s <= val_m:
                    val_m = temp_s
                    p_m = S

            i_prev = I1 + p_prev1
            temp_im = alpha[r_prev, i_prev] - tr_im_ln025
            if temp_im < val_m and not _avoid_insertion_stop(seq, stop_mask_64, phase, int(temp_i[p_prev1]), t, 0):
                val_m = temp_im
                p_m = i_prev

            alpha[r_curr, m_cur] = val_m
            path[t, m_cur] = p_m

        # 2. Forward Insertion States I1..I6
        tr_ii_nt = tr_ii[from_nt, to_nt] if tr_ii[from_nt, to_nt] > -20.0 else LN_025
        tr_mi_nt = tr_mi[from_nt, to_nt] if tr_mi[from_nt, to_nt] > -20.0 else LN_025
        ii_step = tr_ii_pen + tr_ii_nt
        mi_step = tr_mi_pen + tr_mi_nt
        for st_idx in range(6):
            istate = I1 + st_idx
            mstate = M1 + st_idx
            val_i = alpha[r_prev, mstate] - mi_step - (tr_gg if st_idx == 5 else 0.0)
            p_i = mstate
            temp_i[st_idx] = t - 1
            if path[t - 1, istate] == mstate:
                temp_ii = alpha[r_prev, istate] - ii_step
                if temp_ii < val_i:
                    val_i = temp_ii
                    p_i = istate
            alpha[r_curr, istate] = val_i
            path[t, istate] = p_i

        # 3. Reverse Match States M1r..M6r
        rev_stop_tm3 = (t >= 3) and (rev_stop_mask_64[tri_arr[t - 3]] != 0)
        for phase in range(6):
            m_cur = M1r + phase
            if (phase == 2 or phase == 5) and alpha[r_curr, m_cur] >= 1e300:
                continue
            em = em_r[phase, from2, to_nt]
            if (phase == 2 or phase == 5) and em < -5.2:
                em = -5.2
            p_prev1 = (phase + 5) % 6
            p_prev2 = (phase + 4) % 6
            p_prev3 = (phase + 3) % 6
            m_prev1 = M1r + p_prev1
            m_prev2 = M1r + p_prev2
            m_prev3 = M1r + p_prev3
            gg_pen = tr_gg if phase == 0 else 0.0

            if (phase == 0 or phase == 3) and rev_stop_tm3:
                val_mr = alpha[r_prev, Sr] - em
                p_mr = Sr
                if alpha[r_prev, R] - alpha[r_prev, m_prev1] > 8.0:
                    temp_rt = alpha[r_prev, m_prev1] - gg_pen - tr_mm - em + stop_rt_pen
                    if temp_rt < val_mr:
                        val_mr = temp_rt
                        p_mr = m_prev1
                alpha[r_curr, m_cur] = val_mr
                path[t, m_cur] = p_mr
                continue

            val_mr = alpha[r_prev, m_prev1] - gg_pen - tr_mm - em
            p_mr = m_prev1

            temp_d2 = alpha[r_prev, m_prev2] - em - del2_pen
            if temp_d2 < val_mr:
                val_mr = temp_d2
                p_mr = m_prev2

            temp_d3 = alpha[r_prev, m_prev3] - em - del3_pen
            if temp_d3 < val_mr:
                val_mr = temp_d3
                p_mr = m_prev3

            i_prev = I1r + p_prev1
            temp_im = alpha[r_prev, i_prev] - tr_im_ln025
            if temp_im < val_mr and not _avoid_insertion_stop(
                seq, rev_stop_mask_64, phase, int(temp_i_1[p_prev1]), t, 1
            ):
                val_mr = temp_im
                p_mr = i_prev

            if phase == 0 and include_phage and alpha[r_prev, Sr] < 1e300 and path[t - 1, Sr] == Er:
                temp_sr = alpha[r_prev, Sr] - em
                if temp_sr < val_mr:
                    val_mr = temp_sr
                    p_mr = Sr

            alpha[r_curr, m_cur] = val_mr
            path[t, m_cur] = p_mr

        # 4. Reverse Insertion States I1r..I6r
        can_mi_r = (
            (t < 3 or path[t - 3, Sr] != R) and (t < 4 or path[t - 4, Sr] != R) and (t < 5 or path[t - 5, Sr] != R)
        )
        for st_idx in range(6):
            istate = I1r + st_idx
            mstate = M1r + st_idx
            val_ir = np.inf
            p_ir = istate
            if can_mi_r:
                val_ir = alpha[r_prev, mstate] - mi_step - (tr_gg if st_idx == 5 else 0.0)
                p_ir = mstate
                temp_i_1[st_idx] = t - 1
            if path[t - 1, istate] == mstate:
                temp_iir = alpha[r_prev, istate] - ii_step
                if temp_iir < val_ir:
                    val_ir = temp_iir
                    p_ir = istate
            alpha[r_curr, istate] = val_ir
            path[t, istate] = p_ir

        # 5. Non-coding State R
        val_r = alpha[r_prev, R] - tr_rr[cg_bin, from_nt, to_nt] - tr_rr_ln095
        p_r = R
        temp_a = alpha[r_prev, E] - tr_er_ln095
        if temp_a < val_r:
            val_r, p_r = temp_a, E
        temp_a = alpha[r_prev, Er] - tr_er_ln095
        if temp_a < val_r:
            val_r, p_r = temp_a, Er
        alpha[r_curr, R] = val_r
        path[t, R] = p_r

        prev_e_at_curr = alpha[r_curr, E] if alpha[r_curr, E] != 0.0 else np.inf
        prev_er_at_curr = alpha[r_curr, Er] if alpha[r_curr, Er] != 0.0 else np.inf
        tri_t = int(tri_arr[t]) if t < seq_len - 2 else 0

        # 6. End State E (Forward Stop)
        if alpha[r_curr, E] == 0.0:
            alpha[r_curr, E] = np.inf
            path[t, E] = -1
            if t < seq_len - 2 and stop_mask_64[tri_t]:
                alpha[r_next2, E] = np.inf
                for m_end in (M6, M3):
                    temp_a = alpha[r_prev, m_end] - tr_ge
                    if temp_a < alpha[r_next2, E]:
                        alpha[r_next2, E] = temp_a
                        path[t, E] = m_end
                alpha[r_curr, E] = np.inf
                alpha[r_next1, E] = np.inf
                path[t + 1, E] = E
                path[t + 2, E] = E

                for m_rt in (M4, M1):
                    if alpha[r_prev, R] - alpha[r_curr, m_rt] > 8.0:
                        alpha[r_curr, m_rt] += stop_rt_pen
                    else:
                        alpha[r_curr, m_rt] = np.inf

                alpha[r_next2, E] -= stop_log_priors_64[tri_t]
                start_freq = 0.0
                tr_stop_0 = tr_stop[0, cg_bin]
                offset_e = 60 - t
                for i in range(max(0, t - 60), max(0, t - 3) + 1):
                    start_freq -= tr_stop_0[i + offset_e, tri_arr[i]]
                if t < 60:
                    start_freq *= 58.0 / (max(0, t - 3) + 1)
                alpha[r_next2, E] = _modify_border_dist(alpha[r_next2, E], dist_stop[0, cg_bin], start_freq)

        # 7. Start State Sr (Reverse Stop)
        if alpha[r_curr, Sr] == 0.0:
            alpha[r_curr, Sr] = np.inf
            path[t, Sr] = -1
            if t < seq_len - 2 and rev_stop_mask_64[tri_t]:
                alpha[r_curr, Sr] = np.inf
                alpha[r_next1, Sr] = np.inf
                alpha[r_next2, Sr] = alpha[r_prev, R] - tr_rs
                path[t, Sr] = R
                path[t + 1, Sr] = Sr
                path[t + 2, Sr] = Sr

                for e_prev, tr_pen in ((Er, tr_es), (E, tr_es1)):
                    temp_a = alpha[r_prev, e_prev] - tr_pen
                    if temp_a < alpha[r_next2, Sr]:
                        alpha[r_next2, Sr] = temp_a
                        path[t, Sr] = e_prev

                if include_phage:
                    for prev_val, e_prev, tr_pen in ((prev_er_at_curr, Er, tr_es), (prev_e_at_curr, E, tr_es1)):
                        temp_a = prev_val - tr_pen
                        if temp_a < alpha[r_next2, Sr]:
                            alpha[r_next2, Sr] = temp_a
                            path[t, Sr] = e_prev
                    if t + 3 < seq_len and rev_start_mask_64[tri_arr[t + 1]]:
                        temp_a = alpha[r_curr, M6r] - tr_ge - tr_es
                        if temp_a < alpha[r_next2, Sr]:
                            alpha[r_next2, Sr] = temp_a
                            path[t + 2, Sr] = Er

                alpha[r_next2, Sr] -= rev_stop_log_priors_64[tri_t]
                start_freq = 0.0
                if t + 5 < seq_len:
                    tr_start_1 = tr_start[1, cg_bin]
                    offset_sr = -3 - t
                    for i in range(t + 3, min(t + 60, seq_len - 3) + 1):
                        start_freq -= tr_start_1[i + offset_sr, tri_arr[i]]
                alpha[r_next2, Sr] = _modify_border_dist(alpha[r_next2, Sr], dist_start[1, cg_bin], start_freq)

        # 8. Start State S (Forward Start)
        if alpha[r_curr, S] == 0.0:
            alpha[r_curr, S] = np.inf
            path[t, S] = -1
            if t < seq_len - 2 and start_mask_64[tri_t]:
                alpha[r_curr, S] = np.inf
                alpha[r_next1, S] = np.inf
                alpha[r_next2, S] = alpha[r_prev, R] - tr_rs
                path[t, S] = R
                path[t + 1, S] = S
                path[t + 2, S] = S

                for e_prev, tr_pen in ((E, tr_es), (Er, tr_es1)):
                    temp_a = alpha[r_prev, e_prev] - tr_pen
                    if temp_a < alpha[r_next2, S]:
                        alpha[r_next2, S] = temp_a
                        path[t, S] = e_prev

                if include_phage:
                    temp_a = prev_e_at_curr - tr_es
                    if temp_a < alpha[r_next2, S]:
                        alpha[r_next2, S] = temp_a
                        path[t, S] = E
                    if t + 3 < seq_len and stop_mask_64[tri_arr[t + 1]]:
                        temp_a = min(alpha[r_curr, M6], alpha[r_curr, M3]) - tr_ge - tr_es
                        if temp_a < alpha[r_next2, S]:
                            alpha[r_next2, S] = temp_a
                            path[t + 2, S] = E

                alpha[r_next2, S] -= start_log_priors_64[tri_t]
                start_freq = 0.0
                tr_start_0 = tr_start[0, cg_bin]
                offset_s = 30 - t
                for i in range(max(0, t - 30), min(t + 30, seq_len - 3) + 1):
                    start_freq -= tr_start_0[i + offset_s, tri_arr[i]]
                if t < 30:
                    start_freq *= 61.0 / (t + 30 + 1)
                alpha[r_next2, S] = _modify_border_dist(alpha[r_next2, S], dist_start[0, cg_bin], start_freq)

        # 9. End State Er (Reverse Start)
        if alpha[r_curr, Er] == 0.0:
            alpha[r_curr, Er] = np.inf
            path[t, Er] = -1
            if t < seq_len - 2 and rev_start_mask_64[tri_t]:
                alpha[r_curr, Er] = np.inf
                alpha[r_next1, Er] = np.inf
                alpha[r_next2, Er] = alpha[r_prev, M6r] - tr_ge - rev_start_log_priors_64[tri_t]
                path[t, Er] = M6r
                path[t + 1, Er] = Er
                path[t + 2, Er] = Er

                start_freq = 0.0
                tr_stop_1 = tr_stop[1, cg_bin]
                offset_er = 30 - t
                for i in range(max(0, t - 30), min(t + 30, seq_len - 3) + 1):
                    start_freq -= tr_stop_1[i + offset_er, tri_arr[i]]
                if t < 30:
                    start_freq *= 61.0 / (t + 30 + 1)
                alpha[r_next2, Er] = _modify_border_dist(alpha[r_next2, Er], dist_stop[1, cg_bin], start_freq)

        if num_noncoding > 9:
            for i in range(NUM_STATES):
                if i != R:
                    alpha[r_curr, i] = np.inf
                    path[t, i] = R

    last_r = (seq_len - 1) & 3
    best_st = 0
    best_prob = np.inf
    for i in range(NUM_STATES):
        if alpha[last_r, i] < best_prob:
            best_prob = alpha[last_r, i]
            best_st = i
    return best_st, path


@njit(fastmath=True, nogil=True, cache=True)
def _viterbi_backtrack(
    best_st: int,
    path: np.ndarray,
) -> np.ndarray:
    seq_len = len(path)
    vpath = np.zeros(seq_len, dtype=np.int8)
    vpath[seq_len - 1] = best_st
    for t in range(seq_len - 2, -1, -1):
        prev = path[t + 1, vpath[t + 1]]
        vpath[t] = R if prev < 0 else prev
    return vpath
