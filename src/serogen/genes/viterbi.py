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
LN_053 = math.log(0.53)
LN_016 = math.log(0.16)
LN_030 = math.log(0.30)
LN_054 = math.log(0.54)
LN_083 = math.log(0.83)
LN_010 = math.log(0.10)
LN_007 = math.log(0.07)


@njit(fastmath=True, nogil=True, cache=True)
def _modify_border_dist(
    cell_val: float,
    values: np.ndarray,  # type: ignore[type-arg]
    start_freq: float,
) -> float:
    h_kd = values[2] * math.exp(-1.0 * (start_freq - values[1]) ** 2 / (values[0] ** 2) / 2.0)
    r_kd = values[5] * math.exp(-1.0 * (start_freq - values[4]) ** 2 / (values[3] ** 2) / 2.0)
    denom = h_kd + r_kd
    if denom == 0.0:
        ratio = 0.01
    else:
        ratio = h_kd / denom
    if ratio < 0.01:
        ratio = 0.01
    elif ratio > 0.99:
        ratio = 0.99
    return cell_val - math.log(ratio)


@njit(fastmath=True, nogil=True, cache=True)
def _trinucleotide(seq: np.ndarray, i: int) -> int:  # type: ignore[type-arg]
    n0 = seq[i]
    n1 = seq[i + 1]
    n2 = seq[i + 2]
    if n0 > 3 or n1 > 3 or n2 > 3:
        return 0
    return int(n0) * 16 + int(n1) * 4 + int(n2)


@njit(fastmath=True, nogil=True, cache=True)
def _build_local_cg_bins(
    seq: np.ndarray,  # type: ignore[type-arg]
    default_cg_bin: int,
    window: int,
) -> np.ndarray:  # type: ignore[type-arg]
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
def _build_tri_array(seq: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
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


@njit(fastmath=True, nogil=True, cache=True)
def _viterbi_forward(
    seq: np.ndarray,  # type: ignore[type-arg]
    cg_bins: np.ndarray,  # type: ignore[type-arg]
    whole_genome: bool,
    include_phage: bool,
    tr: np.ndarray,  # type: ignore[type-arg]
    tr_mi: np.ndarray,  # type: ignore[type-arg]
    tr_ii: np.ndarray,  # type: ignore[type-arg]
    pi: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_start: np.ndarray,  # type: ignore[type-arg]
    tr_stop: np.ndarray,  # type: ignore[type-arg]
    dist_start: np.ndarray,  # type: ignore[type-arg]
    dist_stop: np.ndarray,  # type: ignore[type-arg]
    start_mask_64: np.ndarray,  # type: ignore[type-arg]
    stop_mask_64: np.ndarray,  # type: ignore[type-arg]
    rev_start_mask_64: np.ndarray,  # type: ignore[type-arg]
    rev_stop_mask_64: np.ndarray,  # type: ignore[type-arg]
    start_log_priors_64: np.ndarray,  # type: ignore[type-arg]
    rev_start_log_priors_64: np.ndarray,  # type: ignore[type-arg]
    stop_log_priors_64: np.ndarray,  # type: ignore[type-arg]
    rev_stop_log_priors_64: np.ndarray,  # type: ignore[type-arg]
) -> tuple[int, np.ndarray]:  # type: ignore[type-arg]
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

    # Initial state costs
    for i in range(NUM_STATES):
        alpha[0, i] = -pi[i]

    if seq_len >= 3:
        tri0 = int(tri_arr[0])
        # Stop codon at start
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

        # Reverse stop codon at start
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
    tr_im_ln025 = tr[TR_IM] + LN_025
    tr_ii_pen = tr[TR_II]
    tr_mi_pen = tr[TR_MI]
    tr_ge = tr[TR_GE]
    tr_er_ln095 = tr[TR_ER] + LN_095
    tr_rr_ln095 = tr[TR_RR] + LN_095
    tr_rs = tr[TR_RS]
    tr_es = tr[TR_ES]
    tr_es1 = tr[TR_ES1]

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

        from2 = from0 * 4 + from_nt
        em_f = e_match[0, cg_bin]
        em_r = e_match[1, cg_bin]

        # Match State M1
        if np.isfinite(alpha[r_curr, M1]):
            em0 = em_f[0, from2, to_nt]
            val_m1 = alpha[r_prev, M6] - tr_gg - tr_mm - em0
            p_m1 = M6

            if not whole_genome:
                for state_from, num_d in (
                    (M5, 2.0),
                    (M4, 3.0),
                    (M3, 4.0),
                    (M2, 5.0),
                    (M1, 6.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em0
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m1:
                        val_m1 = temp_a
                        p_m1 = state_from

            temp_a = alpha[r_prev, S] - em0
            if temp_a <= val_m1:
                val_m1 = temp_a
                p_m1 = S

            temp_a = alpha[r_prev, I6] - tr_im_ln025
            if temp_a < val_m1:
                temp_idx = temp_i[5]
                avoid = False
                if t >= 2 and temp_idx > 0:
                    b0, b1, b2 = seq[temp_idx - 1], seq[temp_idx], seq[t]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m1 = temp_a
                    p_m1 = I6
            alpha[r_curr, M1] = val_m1
            path[t, M1] = p_m1

        # Match State M2
        if np.isfinite(alpha[r_curr, M2]):
            em1 = em_f[1, from2, to_nt]
            val_m2 = alpha[r_prev, M1] - tr_mm - em1
            p_m2 = M1

            if not whole_genome:
                for state_from, num_d in (
                    (M6, 2.0),
                    (M5, 3.0),
                    (M4, 4.0),
                    (M3, 5.0),
                    (M2, 6.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em1
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m2:
                        val_m2 = temp_a
                        p_m2 = state_from

            temp_a = alpha[r_prev, I1] - tr_im_ln025
            if temp_a < val_m2:
                temp_idx = temp_i[0]
                avoid = False
                if t >= 2 and t + 1 < seq_len:
                    b0, b1, b2 = seq[temp_idx], seq[t], seq[t + 1]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m2 = temp_a
                    p_m2 = I1
            alpha[r_curr, M2] = val_m2
            path[t, M2] = p_m2

        # Match State M3
        if np.isfinite(alpha[r_curr, M3]):
            em2 = em_f[2, from2, to_nt]
            val_m3 = alpha[r_prev, M2] - tr_mm - em2
            p_m3 = M2

            if not whole_genome:
                for state_from, num_d in (
                    (M6, 3.0),
                    (M5, 4.0),
                    (M4, 5.0),
                    (M3, 6.0),
                    (M1, 2.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em2
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m3:
                        val_m3 = temp_a
                        p_m3 = state_from

            temp_a = alpha[r_prev, I2] - tr_im_ln025
            if temp_a < val_m3:
                temp_idx = temp_i[1]
                avoid = False
                if t >= 2 and temp_idx > 0:
                    b0, b1, b2 = seq[temp_idx - 1], seq[temp_idx], seq[t]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m3 = temp_a
                    p_m3 = I2
            alpha[r_curr, M3] = val_m3
            path[t, M3] = p_m3

        # Match State M4
        if np.isfinite(alpha[r_curr, M4]):
            em3 = em_f[3, from2, to_nt]
            val_m4 = alpha[r_prev, M3] - tr_mm - em3
            p_m4 = M3

            if not whole_genome:
                for state_from, num_d in (
                    (M6, 4.0),
                    (M5, 5.0),
                    (M4, 6.0),
                    (M2, 2.0),
                    (M1, 3.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em3
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m4:
                        val_m4 = temp_a
                        p_m4 = state_from

            temp_a = alpha[r_prev, I3] - tr_im_ln025
            if temp_a < val_m4:
                val_m4 = temp_a
                p_m4 = I3
            alpha[r_curr, M4] = val_m4
            path[t, M4] = p_m4

        # Match State M5
        if np.isfinite(alpha[r_curr, M5]):
            em4 = em_f[4, from2, to_nt]
            val_m5 = alpha[r_prev, M4] - tr_mm - em4
            p_m5 = M4

            if not whole_genome:
                for state_from, num_d in (
                    (M6, 5.0),
                    (M5, 6.0),
                    (M3, 2.0),
                    (M2, 3.0),
                    (M1, 4.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em4
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m5:
                        val_m5 = temp_a
                        p_m5 = state_from

            temp_a = alpha[r_prev, I4] - tr_im_ln025
            if temp_a < val_m5:
                temp_idx = temp_i[3]
                avoid = False
                if t >= 2 and t + 1 < seq_len:
                    b0, b1, b2 = seq[temp_idx], seq[t], seq[t + 1]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m5 = temp_a
                    p_m5 = I4
            alpha[r_curr, M5] = val_m5
            path[t, M5] = p_m5

        # Match State M6
        if np.isfinite(alpha[r_curr, M6]):
            em5 = em_f[5, from2, to_nt]
            val_m6 = alpha[r_prev, M5] - tr_mm - em5
            p_m6 = M5

            if not whole_genome:
                for state_from, num_d in (
                    (M6, 6.0),
                    (M4, 2.0),
                    (M3, 3.0),
                    (M2, 4.0),
                    (M1, 5.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em5
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m6:
                        val_m6 = temp_a
                        p_m6 = state_from

            temp_a = alpha[r_prev, I5] - tr_im_ln025
            if temp_a < val_m6:
                temp_idx = temp_i[4]
                avoid = False
                if t >= 2 and temp_idx > 0:
                    b0, b1, b2 = seq[temp_idx - 1], seq[temp_idx], seq[t]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m6 = temp_a
                    p_m6 = I5
            alpha[r_curr, M6] = val_m6
            path[t, M6] = p_m6

        # Forward Insertion States I1..I6
        ii_step = tr_ii_pen + tr_ii[from_nt, to_nt]
        mi_step = tr_mi_pen + tr_mi[from_nt, to_nt]
        for st_idx in range(6):
            istate = I1 + st_idx
            mstate = M1 + st_idx
            val_i = alpha[r_prev, istate] - ii_step
            p_i = istate
            last_i = tr_gg if st_idx == 5 else 0.0
            temp_a = alpha[r_prev, mstate] - mi_step - last_i
            if temp_a < val_i:
                val_i = temp_a
                p_i = mstate
                temp_i[st_idx] = t - 1
            alpha[r_curr, istate] = val_i
            path[t, istate] = p_i

        # Reverse Match State M1r & M4r share rev_stop_mask_64[tri_arr[t - 3]]
        rev_stop_tm3 = (t >= 3) and (rev_stop_mask_64[tri_arr[t - 3]] != 0)

        em0r = em_r[0, from2, to_nt]
        if rev_stop_tm3:
            alpha[r_curr, M1r] = alpha[r_prev, Sr] - em0r
            path[t, M1r] = Sr
        else:
            val_m1r = alpha[r_prev, M6r] - tr_gg - tr_mm - em0r
            p_m1r = M6r

            if not whole_genome:
                for state_from, num_d in (
                    (M5r, 2.0),
                    (M4r, 3.0),
                    (M3r, 4.0),
                    (M2r, 5.0),
                    (M1r, 6.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em0r
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m1r:
                        val_m1r = temp_a
                        p_m1r = state_from

            temp_a = alpha[r_prev, I6r] - tr_im_ln025
            if temp_a < val_m1r:
                temp_idx = temp_i_1[5]
                avoid = False
                if t >= 2 and temp_idx > 1:
                    b0, b1, b2 = seq[temp_idx - 1], seq[temp_idx], seq[t]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and rev_stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m1r = temp_a
                    p_m1r = I6r

            if include_phage and np.isfinite(alpha[r_prev, Sr]) and path[t - 1, Sr] == Er:
                temp_a = alpha[r_prev, Sr] - em0r
                if temp_a < val_m1r:
                    val_m1r = temp_a
                    p_m1r = Sr
            alpha[r_curr, M1r] = val_m1r
            path[t, M1r] = p_m1r

        # Reverse Match State M2r
        em1r = em_r[1, from2, to_nt]
        val_m2r = alpha[r_prev, M1r] - tr_mm - em1r
        p_m2r = M1r

        if not whole_genome:
            for state_from, num_d in (
                (M6r, 2.0),
                (M5r, 3.0),
                (M4r, 4.0),
                (M3r, 5.0),
                (M2r, 6.0),
            ):
                temp_a = (
                    alpha[r_prev, state_from]
                    - tr[TR_MD]
                    - em1r
                    - LN_025 * (num_d - 1.0)
                    - tr[TR_DD] * (num_d - 2.0)
                    - tr[TR_DM]
                )
                if temp_a < val_m2r:
                    val_m2r = temp_a
                    p_m2r = state_from

        temp_a = alpha[r_prev, I1r] - tr_im_ln025
        if temp_a < val_m2r:
            temp_idx = temp_i_1[0]
            avoid = False
            if t >= 2 and t + 1 < seq_len:
                b0, b1, b2 = seq[temp_idx], seq[t], seq[t + 1]
                if b0 <= 3 and b1 <= 3 and b2 <= 3 and rev_stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                    avoid = True
            if not avoid:
                val_m2r = temp_a
                p_m2r = I1r
        alpha[r_curr, M2r] = val_m2r
        path[t, M2r] = p_m2r

        # Reverse Match State M3r
        if np.isfinite(alpha[r_curr, M3r]):
            em2r = em_r[2, from2, to_nt]
            val_m3r = alpha[r_prev, M2r] - tr_mm - em2r
            p_m3r = M2r

            if not whole_genome:
                for state_from, num_d in (
                    (M6r, 3.0),
                    (M5r, 4.0),
                    (M4r, 5.0),
                    (M3r, 6.0),
                    (M1r, 2.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em2r
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m3r:
                        val_m3r = temp_a
                        p_m3r = state_from

            temp_a = alpha[r_prev, I2r] - tr_im_ln025
            if temp_a < val_m3r:
                temp_idx = temp_i_1[1]
                avoid = False
                if t >= 2 and temp_idx > 1:
                    b0, b1, b2 = seq[temp_idx - 1], seq[temp_idx], seq[t]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and rev_stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m3r = temp_a
                    p_m3r = I2r
            alpha[r_curr, M3r] = val_m3r
            path[t, M3r] = p_m3r

        # Reverse Match State M4r
        em3r = em_r[3, from2, to_nt]
        if rev_stop_tm3:
            alpha[r_curr, M4r] = alpha[r_prev, Sr] - em3r
            path[t, M4r] = Sr
        else:
            val_m4r = alpha[r_prev, M3r] - tr_mm - em3r
            p_m4r = M3r

            if not whole_genome:
                for state_from, num_d in (
                    (M6r, 4.0),
                    (M5r, 5.0),
                    (M4r, 6.0),
                    (M2r, 2.0),
                    (M1r, 3.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em3r
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m4r:
                        val_m4r = temp_a
                        p_m4r = state_from

            temp_a = alpha[r_prev, I3r] - tr_im_ln025
            if temp_a < val_m4r:
                val_m4r = temp_a
                p_m4r = I3r
            alpha[r_curr, M4r] = val_m4r
            path[t, M4r] = p_m4r

        # Reverse Match State M5r
        em4r = em_r[4, from2, to_nt]
        val_m5r = alpha[r_prev, M4r] - tr_mm - em4r
        p_m5r = M4r

        if not whole_genome:
            for state_from, num_d in (
                (M6r, 5.0),
                (M5r, 6.0),
                (M3r, 2.0),
                (M2r, 3.0),
                (M1r, 4.0),
            ):
                temp_a = (
                    alpha[r_prev, state_from]
                    - tr[TR_MD]
                    - em4r
                    - LN_025 * (num_d - 1.0)
                    - tr[TR_DD] * (num_d - 2.0)
                    - tr[TR_DM]
                )
                if temp_a < val_m5r:
                    val_m5r = temp_a
                    p_m5r = state_from

        temp_a = alpha[r_prev, I4r] - tr_im_ln025
        if temp_a < val_m5r:
            temp_idx = temp_i_1[3]
            avoid = False
            if t >= 2 and t + 1 < seq_len:
                b0, b1, b2 = seq[temp_idx], seq[t], seq[t + 1]
                if b0 <= 3 and b1 <= 3 and b2 <= 3 and rev_stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                    avoid = True
            if not avoid:
                val_m5r = temp_a
                p_m5r = I4r
        alpha[r_curr, M5r] = val_m5r
        path[t, M5r] = p_m5r

        # Reverse Match State M6r
        if np.isfinite(alpha[r_curr, M6r]):
            em5r = em_r[5, from2, to_nt]
            val_m6r = alpha[r_prev, M5r] - tr_mm - em5r
            p_m6r = M5r

            if not whole_genome:
                for state_from, num_d in (
                    (M6r, 6.0),
                    (M4r, 2.0),
                    (M3r, 3.0),
                    (M2r, 4.0),
                    (M1r, 5.0),
                ):
                    temp_a = (
                        alpha[r_prev, state_from]
                        - tr[TR_MD]
                        - em5r
                        - LN_025 * (num_d - 1.0)
                        - tr[TR_DD] * (num_d - 2.0)
                        - tr[TR_DM]
                    )
                    if temp_a < val_m6r:
                        val_m6r = temp_a
                        p_m6r = state_from

            temp_a = alpha[r_prev, I5r] - tr_im_ln025
            if temp_a < val_m6r:
                temp_idx = temp_i_1[4]
                avoid = False
                if t >= 2 and temp_idx > 1:
                    b0, b1, b2 = seq[temp_idx - 1], seq[temp_idx], seq[t]
                    if b0 <= 3 and b1 <= 3 and b2 <= 3 and rev_stop_mask_64[(int(b0) << 4) | (int(b1) << 2) | int(b2)]:
                        avoid = True
                if not avoid:
                    val_m6r = temp_a
                    p_m6r = I5r
            alpha[r_curr, M6r] = val_m6r
            path[t, M6r] = p_m6r

        # Reverse Insertion States I1r..I6r
        can_mi_r = (
            (t < 3 or path[t - 3, Sr] != R) and (t < 4 or path[t - 4, Sr] != R) and (t < 5 or path[t - 5, Sr] != R)
        )
        for st_idx in range(6):
            istate = I1r + st_idx
            val_ir = alpha[r_prev, istate] - ii_step
            p_ir = istate
            if can_mi_r:
                mstate = M1r + st_idx
                last_i = tr_gg if st_idx == 5 else 0.0
                temp_a = alpha[r_prev, mstate] - mi_step - last_i
                if temp_a < val_ir:
                    val_ir = temp_a
                    p_ir = mstate
                    temp_i_1[st_idx] = t - 1
            alpha[r_curr, istate] = val_ir
            path[t, istate] = p_ir

        # Non-coding State R
        val_r = alpha[r_prev, R] - tr_rr[cg_bin, from_nt, to_nt] - tr_rr_ln095
        p_r = R

        temp_a = alpha[r_prev, E] - tr_er_ln095
        if temp_a < val_r:
            val_r = temp_a
            p_r = E

        temp_a = alpha[r_prev, Er] - tr_er_ln095
        if temp_a < val_r:
            val_r = temp_a
            p_r = Er
        alpha[r_curr, R] = val_r
        path[t, R] = p_r

        # Preserve E and Er scores ending at t (for 1-bp overlapping operons TGATG / CATCA)
        prev_e_at_curr = alpha[r_curr, E] if alpha[r_curr, E] != 0.0 else np.inf
        prev_er_at_curr = alpha[r_curr, Er] if alpha[r_curr, Er] != 0.0 else np.inf

        tri_t = int(tri_arr[t]) if t < seq_len - 2 else 0

        # End State E (Forward Stop)
        if alpha[r_curr, E] == 0.0:
            alpha[r_curr, E] = np.inf
            path[t, E] = -1

            if t < seq_len - 2 and stop_mask_64[tri_t]:
                alpha[r_next2, E] = np.inf

                temp_a = alpha[r_prev, M6] - tr_ge
                if temp_a < alpha[r_next2, E]:
                    alpha[r_next2, E] = temp_a
                    path[t, E] = M6

                temp_a = alpha[r_prev, M3] - tr_ge
                if temp_a < alpha[r_next2, E]:
                    alpha[r_next2, E] = temp_a
                    path[t, E] = M3

                alpha[r_curr, E] = np.inf
                alpha[r_next1, E] = np.inf
                path[t + 1, E] = E
                path[t + 2, E] = E

                alpha[r_next2, M6] = np.inf
                alpha[r_next1, M5] = np.inf
                alpha[r_curr, M4] = np.inf
                alpha[r_next2, M3] = np.inf
                alpha[r_next1, M2] = np.inf
                alpha[r_curr, M1] = np.inf

                alpha[r_next2, E] -= stop_log_priors_64[tri_t]

                start_freq = 0.0
                start_i = max(0, t - 60)
                end_i = max(0, t - 3)
                tr_stop_0 = tr_stop[0, cg_bin]
                offset_e = 60 - t
                for i in range(start_i, end_i + 1):
                    start_freq -= tr_stop_0[i + offset_e, tri_arr[i]]
                if t < 60:
                    denom = max(0, t - 3) + 1
                    start_freq *= 58.0 / denom

                alpha[r_next2, E] = _modify_border_dist(alpha[r_next2, E], dist_stop[0, cg_bin], start_freq)

        # Start State Sr (Reverse Stop)
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

                temp_a = alpha[r_prev, Er] - tr_es
                if temp_a < alpha[r_next2, Sr]:
                    alpha[r_next2, Sr] = temp_a
                    path[t, Sr] = Er

                temp_a = alpha[r_prev, E] - tr_es1
                if temp_a < alpha[r_next2, Sr]:
                    alpha[r_next2, Sr] = temp_a
                    path[t, Sr] = E

                if include_phage:
                    # 1-bp co-directional overlap on reverse strand (CATCA)
                    temp_a = prev_er_at_curr - tr_es
                    if temp_a < alpha[r_next2, Sr]:
                        alpha[r_next2, Sr] = temp_a
                        path[t, Sr] = Er
                    # 1-bp convergent overlap (+1 E with -1 Sr)
                    temp_a = prev_e_at_curr - tr_es1
                    if temp_a < alpha[r_next2, Sr]:
                        alpha[r_next2, Sr] = temp_a
                        path[t, Sr] = E
                    # 4-bp co-directional overlap on reverse strand (TCAT)
                    if t + 3 < seq_len and rev_start_mask_64[tri_arr[t + 1]]:
                        temp_a = alpha[r_curr, M6r] - tr_ge - tr_es
                        if temp_a < alpha[r_next2, Sr]:
                            alpha[r_next2, Sr] = temp_a
                            path[t + 2, Sr] = Er

                alpha[r_next2, M3r] = np.inf
                alpha[r_next2, M6r] = np.inf

                alpha[r_next2, Sr] -= rev_stop_log_priors_64[tri_t]

                start_freq = 0.0
                if t + 5 < seq_len:
                    start_i = t + 3
                    end_i = min(t + 60, seq_len - 3)
                    tr_start_1 = tr_start[1, cg_bin]
                    offset_sr = -3 - t
                    for i in range(start_i, end_i + 1):
                        start_freq -= tr_start_1[i + offset_sr, tri_arr[i]]

                alpha[r_next2, Sr] = _modify_border_dist(alpha[r_next2, Sr], dist_start[1, cg_bin], start_freq)

        # Start State S (Forward Start)
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

                temp_a = alpha[r_prev, E] - tr_es
                if temp_a < alpha[r_next2, S]:
                    alpha[r_next2, S] = temp_a
                    path[t, S] = E

                temp_a = alpha[r_prev, Er] - tr_es1
                if temp_a < alpha[r_next2, S]:
                    alpha[r_next2, S] = temp_a
                    path[t, S] = Er

                if include_phage:
                    # 1-bp co-directional overlap on forward strand (TGATG / TAATG)
                    temp_a = prev_e_at_curr - tr_es
                    if temp_a < alpha[r_next2, S]:
                        alpha[r_next2, S] = temp_a
                        path[t, S] = E
                    # 4-bp co-directional overlap on forward strand (ATGA)
                    if t + 3 < seq_len and stop_mask_64[tri_arr[t + 1]]:
                        best_m = min(alpha[r_curr, M6], alpha[r_curr, M3])
                        temp_a = best_m - tr_ge - tr_es
                        if temp_a < alpha[r_next2, S]:
                            alpha[r_next2, S] = temp_a
                            path[t + 2, S] = E

                alpha[r_next2, S] -= start_log_priors_64[tri_t]

                start_freq = 0.0
                start_i = max(0, t - 30)
                end_i = min(t + 30, seq_len - 3)
                tr_start_0 = tr_start[0, cg_bin]
                offset_s = 30 - t
                for i in range(start_i, end_i + 1):
                    start_freq -= tr_start_0[i + offset_s, tri_arr[i]]
                if t < 30:
                    start_freq *= 61.0 / (t + 30 + 1)

                alpha[r_next2, S] = _modify_border_dist(alpha[r_next2, S], dist_start[0, cg_bin], start_freq)

        # End State Er (Reverse Start)
        if alpha[r_curr, Er] == 0.0:
            alpha[r_curr, Er] = np.inf
            path[t, Er] = -1

            if t < seq_len - 2 and rev_start_mask_64[tri_t]:
                alpha[r_curr, Er] = np.inf
                alpha[r_next1, Er] = np.inf
                alpha[r_next2, Er] = alpha[r_prev, M6r] - tr_ge
                path[t, Er] = M6r
                path[t + 1, Er] = Er
                path[t + 2, Er] = Er

                alpha[r_next2, Er] -= rev_start_log_priors_64[tri_t]

                start_freq = 0.0
                start_i = max(0, t - 30)
                end_i = min(t + 30, seq_len - 3)
                tr_stop_1 = tr_stop[1, cg_bin]
                offset_er = 30 - t
                for i in range(start_i, end_i + 1):
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
    path: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    seq_len = len(path)
    vpath = np.zeros(seq_len, dtype=np.int8)

    vpath[seq_len - 1] = best_st
    for t in range(seq_len - 2, -1, -1):
        prev = path[t + 1, vpath[t + 1]]
        if prev < 0:
            prev = R
        vpath[t] = prev

    return vpath
