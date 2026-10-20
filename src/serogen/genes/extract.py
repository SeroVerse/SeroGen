r"""Numba-accelerated gene coordinate extraction, frameshift indel tracking, and parallel window decoding."""

import numpy as np
from numba import njit, prange

from .viterbi import (
    I1,
    I6,
    M1,
    M4,
    M6,
    E,
    Er,
    I1r,
    I6r,
    M1r,
    M4r,
    M6r,
    R,
    S,
    Sr,
    _build_local_cg_bins,
    _build_tri_array,
    _modify_border_dist,
    _viterbi_backtrack,
    _viterbi_forward,
)


@njit(cache=True, nogil=True)
def _build_window_table(
    contig_offsets: np.ndarray,  # type: ignore[type-arg]
    contig_lengths: np.ndarray,  # type: ignore[type-arg]
    window_size: int,
    overlap: int,
    min_contig_len: int,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    np.ndarray[tuple[int], np.dtype[np.int64]],
]:
    r"""Build a flat descriptor table of overlapping windows across all contigs in a batch."""
    n_contigs = len(contig_lengths)
    step = window_size - overlap if window_size > overlap and window_size > 0 else window_size
    margin = overlap // 2 if window_size > overlap and window_size > 0 else 0

    # Pass 1: count total windows across all contigs
    n_windows = 0
    for c_idx in range(n_contigs):
        c_len = int(contig_lengths[c_idx])
        if c_len < min_contig_len:
            continue
        if window_size <= 0 or c_len <= window_size:
            n_windows += 1
        else:
            for _ in range(0, c_len, step):
                n_windows += 1

    win_contig_idx = np.empty(n_windows, dtype=np.uint32)
    win_buf_starts = np.empty(n_windows, dtype=np.int64)
    win_buf_ends = np.empty(n_windows, dtype=np.int64)
    win_c_starts = np.empty(n_windows, dtype=np.int32)
    win_left_cutoffs = np.empty(n_windows, dtype=np.int32)
    win_right_cutoffs = np.empty(n_windows, dtype=np.int32)
    win_is_left_edge = np.empty(n_windows, dtype=np.bool_)
    win_is_right_edge = np.empty(n_windows, dtype=np.bool_)
    win_vp_offsets = np.zeros(n_windows + 1, dtype=np.int64)

    if n_windows == 0:
        return (
            win_contig_idx,
            win_buf_starts,
            win_buf_ends,
            win_c_starts,
            win_left_cutoffs,
            win_right_cutoffs,
            win_is_left_edge,
            win_is_right_edge,
            win_vp_offsets,
        )

    # Pass 2: populate window descriptors
    w_i = 0
    vp_acc = np.int64(0)
    for c_idx in range(n_contigs):
        c_len = int(contig_lengths[c_idx])
        if c_len < min_contig_len:
            continue
        c_off = np.int64(contig_offsets[c_idx])
        if window_size <= 0 or c_len <= window_size:
            win_contig_idx[w_i] = c_idx
            win_buf_starts[w_i] = c_off
            win_buf_ends[w_i] = c_off + c_len
            win_c_starts[w_i] = 0
            win_left_cutoffs[w_i] = 0
            win_right_cutoffs[w_i] = c_len
            win_is_left_edge[w_i] = True
            win_is_right_edge[w_i] = True
            win_vp_offsets[w_i] = vp_acc
            vp_acc += c_len
            w_i += 1
        else:
            for w_start in range(0, c_len, step):
                w_end = min(w_start + window_size, c_len)
                w_len = w_end - w_start
                win_contig_idx[w_i] = c_idx
                win_buf_starts[w_i] = c_off + w_start
                win_buf_ends[w_i] = c_off + w_end
                win_c_starts[w_i] = w_start
                win_left_cutoffs[w_i] = 0 if w_start == 0 else w_start + margin
                win_right_cutoffs[w_i] = c_len if w_end == c_len else w_end - margin
                win_is_left_edge[w_i] = w_start == 0
                win_is_right_edge[w_i] = w_end == c_len
                win_vp_offsets[w_i] = vp_acc
                vp_acc += w_len
                w_i += 1

    win_vp_offsets[n_windows] = vp_acc
    return (
        win_contig_idx,
        win_buf_starts,
        win_buf_ends,
        win_c_starts,
        win_left_cutoffs,
        win_right_cutoffs,
        win_is_left_edge,
        win_is_right_edge,
        win_vp_offsets,
    )


@njit(fastmath=True, nogil=True, cache=True)
def _score_candidate_orf(
    seq: np.ndarray,  # type: ignore[type-arg]
    tri_arr: np.ndarray,  # type: ignore[type-arg]
    s: int,
    e: int,
    strand: int,
    cg_bins: np.ndarray,  # type: ignore[type-arg]
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
    tr_start: np.ndarray,  # type: ignore[type-arg]
    tr_stop: np.ndarray,  # type: ignore[type-arg]
    dist_start: np.ndarray,  # type: ignore[type-arg]
    dist_stop: np.ndarray,  # type: ignore[type-arg]
    start_log_priors_64: np.ndarray,  # type: ignore[type-arg]
    rev_start_log_priors_64: np.ndarray,  # type: ignore[type-arg]
    stop_log_priors_64: np.ndarray,  # type: ignore[type-arg]
    rev_stop_log_priors_64: np.ndarray,  # type: ignore[type-arg]
) -> tuple[float, float, float]:
    r"""Evaluate HMM log-likelihood ratio and start/stop boundary scores for a candidate ORF."""
    seq_len = len(seq)
    if s < 35 or e + 35 >= seq_len or e - s < 60:
        return -1e9, 1e9, 1e9
    cg_b = int(cg_bins[(s + e) // 2])
    ll_diff = 0.0
    st_idx = 0 if strand == 1 else 1
    for idx in range(e - s - 6):
        t = s + 3 + idx
        b0, b1, b2 = int(seq[t - 2]), int(seq[t - 1]), int(seq[t])
        if b0 > 3 or b1 > 3 or b2 > 3:
            continue
        from2 = b0 * 4 + b1
        ll_diff += e_match[st_idx, cg_b, idx % 6, from2, b2] - tr_rr[cg_b, b1, b2]

    if strand == 1:
        tri_s = int(tri_arr[s])
        tri_e = int(tri_arr[e - 3])
        f_s = 0.0
        for j in range(61):
            f_s -= tr_start[0, cg_b, j, tri_arr[s - 30 + j]]
        b_s = _modify_border_dist(-start_log_priors_64[tri_s], dist_start[0, cg_b], f_s)
        f_e = 0.0
        s_i = max(0, e - 63)
        for i in range(s_i, e - 5):
            f_e -= tr_stop[0, cg_b, i + 60 - (e - 3), tri_arr[i]]
        b_e = _modify_border_dist(-stop_log_priors_64[tri_e], dist_stop[0, cg_b], f_e)
    else:
        tri_sr = int(tri_arr[s])
        tri_er = int(tri_arr[e - 3])
        f_er = 0.0
        for j in range(61):
            f_er -= tr_stop[1, cg_b, j, tri_arr[e - 33 + j]]
        b_s = _modify_border_dist(-rev_start_log_priors_64[tri_er], dist_stop[1, cg_b], f_er)
        f_sr = 0.0
        e_i = min(s + 60, seq_len - 3)
        for i in range(s + 3, e_i + 1):
            f_sr -= tr_start[1, cg_b, i - 3 - s, tri_arr[i]]
        b_e = _modify_border_dist(-rev_stop_log_priors_64[tri_sr], dist_start[1, cg_b], f_sr)

    return ll_diff, b_s, b_e


@njit(fastmath=True, nogil=True, cache=True)
def _find_best_gap_orf(
    seq: np.ndarray,  # type: ignore[type-arg]
    tri_arr: np.ndarray,  # type: ignore[type-arg]
    prev_e: int,
    curr_s: int,
    min_gene_len: int,
    cg_bins: np.ndarray,  # type: ignore[type-arg]
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
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
) -> tuple[int, int, int]:
    r"""Rescue overlapping or short phage ORFs sitting in uncalled gaps between primary Viterbi genes."""
    seq_len = len(seq)
    if curr_s - prev_e < 60:
        return -1, -1, 0
    win_lo = max(35, prev_e - 66)
    win_hi = min(seq_len - 35, curr_s + 36)
    if win_hi - win_lo < min_gene_len:
        return -1, -1, 0

    best_score = 1.5
    best_s = -1
    best_e = -1
    best_st = 0

    # Scan forward (+1) stop codons in [win_lo + min_gene_len - 3 .. win_hi - 3]
    fwd_stop_lo = max(win_lo + min_gene_len - 3, prev_e + 57)
    for stop_pos in range(fwd_stop_lo, win_hi - 2):
        tri_e = int(tri_arr[stop_pos])
        if not stop_mask_64[tri_e]:
            continue
        e_cand = stop_pos + 3
        if e_cand > curr_s + 36:
            continue
        best_stop_net = -1e9
        best_stop_sel = -1e9
        best_s_for_stop = -1
        pos = stop_pos - 3
        while pos >= win_lo:
            tri_s = int(tri_arr[pos])
            if stop_mask_64[tri_s]:
                break
            cand_len = e_cand - pos
            if cand_len >= min_gene_len and start_mask_64[tri_s]:
                max_left_ov = 66 if cand_len >= 240 else 36
                if prev_e - pos <= max_left_ov:
                    ov_lo = max(pos, prev_e)
                    ov_hi = min(e_cand, curr_s)
                    if ov_hi - ov_lo >= 60:
                        lld, bs, be = _score_candidate_orf(
                            seq,
                            tri_arr,
                            pos,
                            e_cand,
                            1,
                            cg_bins,
                            e_match,
                            tr_rr,
                            tr_start,
                            tr_stop,
                            dist_start,
                            dist_stop,
                            start_log_priors_64,
                            rev_start_log_priors_64,
                            stop_log_priors_64,
                            rev_stop_log_priors_64,
                        )
                        net = lld - bs - be
                        valid = False
                        if bs < 0.85 and net > 1.5:
                            valid = True
                        elif bs < 0.30 and be < 2.2 and lld > -1.0:
                            valid = True
                            if net < 2.0:
                                net = 2.0
                        elif cand_len >= 240 and lld > 15.0 and net > 8.0:
                            valid = True
                        elif cand_len >= 300 and bs < 0.42 and be < 2.8 and lld > -6.0:
                            valid = True
                            if net < 1.8:
                                net = 1.8
                        elif curr_s - prev_e >= 250 and bs < 0.70 and be < 3.5 and lld > -2.0:
                            valid = True
                            if net < 1.6:
                                net = 1.6
                        if valid:
                            sel = lld - 1.25 * bs - be
                            if sel > best_stop_sel:
                                best_stop_sel = sel
                                best_stop_net = net
                                best_s_for_stop = pos
            pos -= 3
        if best_s_for_stop >= 0 and best_stop_net > best_score:
            best_score = best_stop_net
            best_s = best_s_for_stop
            best_e = e_cand
            best_st = 1

    # Scan reverse (-1) stop codons in [win_lo .. win_hi - min_gene_len]
    rev_stop_hi = min(win_hi - min_gene_len, curr_s - 60)
    for stop_pos in range(win_lo, rev_stop_hi + 1):
        tri_sr = int(tri_arr[stop_pos])
        if not rev_stop_mask_64[tri_sr]:
            continue
        s_cand = stop_pos
        best_stop_net = -1e9
        best_stop_sel = -1e9
        best_e_for_stop = -1
        pos = stop_pos + 3
        while pos + 3 <= win_hi:
            tri_er = int(tri_arr[pos])
            if rev_stop_mask_64[tri_er]:
                break
            e_cand = pos + 3
            if e_cand - curr_s > 36:
                break
            cand_len = e_cand - s_cand
            if cand_len >= min_gene_len and rev_start_mask_64[tri_er]:
                max_left_ov = 66 if cand_len >= 240 else 36
                if prev_e - s_cand <= max_left_ov:
                    ov_lo = max(s_cand, prev_e)
                    ov_hi = min(e_cand, curr_s)
                    if ov_hi - ov_lo >= 60:
                        lld, bs, be = _score_candidate_orf(
                            seq,
                            tri_arr,
                            s_cand,
                            e_cand,
                            -1,
                            cg_bins,
                            e_match,
                            tr_rr,
                            tr_start,
                            tr_stop,
                            dist_start,
                            dist_stop,
                            start_log_priors_64,
                            rev_start_log_priors_64,
                            stop_log_priors_64,
                            rev_stop_log_priors_64,
                        )
                        net = lld - bs - be
                        valid = False
                        if bs < 0.85 and net > 1.5:
                            valid = True
                        elif bs < 0.30 and be < 2.2 and lld > -1.0:
                            valid = True
                            if net < 2.0:
                                net = 2.0
                        elif cand_len >= 240 and lld > 15.0 and net > 8.0:
                            valid = True
                        elif cand_len >= 300 and bs < 0.42 and be < 2.8 and lld > -6.0:
                            valid = True
                            if net < 1.8:
                                net = 1.8
                        elif curr_s - prev_e >= 250 and bs < 0.70 and be < 3.5 and lld > -2.0:
                            valid = True
                            if net < 1.6:
                                net = 1.6
                        if valid:
                            sel = lld - 1.25 * bs - be
                            if sel > best_stop_sel:
                                best_stop_sel = sel
                                best_stop_net = net
                                best_e_for_stop = e_cand
            pos += 3
        if best_e_for_stop >= 0 and best_stop_net > best_score:
            best_score = best_stop_net
            best_s = s_cand
            best_e = best_e_for_stop
            best_st = -1

    return best_s, best_e, best_st


@njit(fastmath=True, nogil=True, cache=True)
def _is_shadowed_by_opposite_orf(
    seq: np.ndarray,  # type: ignore[type-arg]
    tri_arr: np.ndarray,  # type: ignore[type-arg]
    prev_e: int,
    local_s: int,
    local_e: int,
    strand: int,
    cg_bins: np.ndarray,  # type: ignore[type-arg]
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
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
) -> bool:
    r"""Check if a short, negative-net Viterbi call is a shadow masking a stronger opposite-strand ORF."""
    if local_e - local_s > 300:
        return False
    lld_c, bs_c, be_c = _score_candidate_orf(
        seq,
        tri_arr,
        local_s,
        local_e,
        strand,
        cg_bins,
        e_match,
        tr_rr,
        tr_start,
        tr_stop,
        dist_start,
        dist_stop,
        start_log_priors_64,
        rev_start_log_priors_64,
        stop_log_priors_64,
        rev_stop_log_priors_64,
    )
    net_c = lld_c - bs_c - be_c
    if not (net_c < -0.5 and (lld_c < 0.0 or bs_c > 3.5 or be_c > 4.5)):
        return False

    seq_len = len(seq)
    win_lo = max(35, prev_e - 66)
    win_hi = min(seq_len - 35, local_e + 360)
    if strand == -1:
        # Check for a strong +1 ORF overlapping [local_s, local_e]
        for stop_pos in range(max(win_lo + 177, local_s + 30), win_hi - 2):
            if not stop_mask_64[tri_arr[stop_pos]]:
                continue
            e_cand = stop_pos + 3
            pos = stop_pos - 3
            while pos >= win_lo:
                tri_s = int(tri_arr[pos])
                if stop_mask_64[tri_s]:
                    break
                if e_cand - pos >= 180 and start_mask_64[tri_s]:
                    if min(e_cand, local_e) - max(pos, local_s) >= 24:
                        lld, bs, be = _score_candidate_orf(
                            seq,
                            tri_arr,
                            pos,
                            e_cand,
                            1,
                            cg_bins,
                            e_match,
                            tr_rr,
                            tr_start,
                            tr_stop,
                            dist_start,
                            dist_stop,
                            start_log_priors_64,
                            rev_start_log_priors_64,
                            stop_log_priors_64,
                            rev_stop_log_priors_64,
                        )
                        if lld - bs - be > 5.0 or (e_cand - pos >= 300 and bs < 0.42 and be < 2.8 and lld > -6.0):
                            return True
                pos -= 3
    else:
        # Check for a strong -1 ORF overlapping [local_s, local_e]
        for stop_pos in range(win_lo, min(local_e - 30, win_hi - 180) + 1):
            if not rev_stop_mask_64[tri_arr[stop_pos]]:
                continue
            s_cand = stop_pos
            pos = stop_pos + 3
            while pos + 3 <= win_hi:
                tri_er = int(tri_arr[pos])
                if rev_stop_mask_64[tri_er]:
                    break
                e_cand = pos + 3
                if e_cand - s_cand >= 180 and rev_start_mask_64[tri_er]:
                    if min(e_cand, local_e) - max(s_cand, local_s) >= 24:
                        lld, bs, be = _score_candidate_orf(
                            seq,
                            tri_arr,
                            s_cand,
                            e_cand,
                            -1,
                            cg_bins,
                            e_match,
                            tr_rr,
                            tr_start,
                            tr_stop,
                            dist_start,
                            dist_stop,
                            start_log_priors_64,
                            rev_start_log_priors_64,
                            stop_log_priors_64,
                            rev_stop_log_priors_64,
                        )
                        if lld - bs - be > 5.0 or (e_cand - s_cand >= 300 and bs < 0.42 and be < 2.8 and lld > -6.0):
                            return True
                pos += 3
    return False


@njit(fastmath=True, nogil=True, cache=True)
def _scan_window_genes(
    raw_seq: np.ndarray,  # type: ignore[type-arg]
    seq: np.ndarray,  # type: ignore[type-arg]
    vpath: np.ndarray,  # type: ignore[type-arg]
    cg_bins: np.ndarray,  # type: ignore[type-arg]
    whole_genome: bool,
    include_phage: bool,
    min_gene_len: int,
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
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
    lut_rc: np.ndarray,  # type: ignore[type-arg]
    contig_idx: int,
    c_start: int,
    left_cutoff: int,
    right_cutoff: int,
    is_contig_left: bool,
    is_contig_right: bool,
    write: bool,
    out_starts: np.ndarray,  # type: ignore[type-arg]
    out_ends: np.ndarray,  # type: ignore[type-arg]
    out_strands: np.ndarray,  # type: ignore[type-arg]
    out_partials: np.ndarray,  # type: ignore[type-arg]
    out_contig_indices: np.ndarray,  # type: ignore[type-arg]
    out_cds_seqs: np.ndarray,  # type: ignore[type-arg]
    out_cds_offsets: np.ndarray,  # type: ignore[type-arg]
    out_cds_lengths: np.ndarray,  # type: ignore[type-arg]
    out_indel_pos: np.ndarray,  # type: ignore[type-arg]
    out_indel_kinds: np.ndarray,  # type: ignore[type-arg]
    out_indel_offsets: np.ndarray,  # type: ignore[type-arg]
    out_indel_lengths: np.ndarray,  # type: ignore[type-arg]
    gene_off: int,
    cds_off: int,
    indel_off: int,
) -> tuple[int, int, int]:
    r"""Scan a single decoded window to count (`write=False`) or populate (`write=True`) SoA gene records."""
    seq_len = len(seq)
    tri_arr = _build_tri_array(seq)
    gene_len = min_gene_len if whole_genome else min(min_gene_len, 60)

    dna_buf = np.empty(seq_len * 2 + 100 if write else 0, dtype=np.uint8)
    indel_pos_buf = np.empty(seq_len + 16 if write else 0, dtype=np.int32)
    indel_kind_buf = np.empty(seq_len + 16 if write else 0, dtype=np.int8)

    dna_len = 0
    curr_indel_len = 0
    codon_start = 0
    start_t = -1
    dna_start_t_withstop = 0
    dna_start_t = 0
    prev_match = S
    prev_local_e = 0

    n_genes = 0
    total_cds = 0
    total_indels = 0

    for t in range(seq_len):
        st = vpath[t]

        if codon_start == 0 and start_t < 0 and ((M1 <= st <= M6) or (M1r <= st <= M6r) or st == S or st == Sr):
            dna_start_t_withstop = t + 1
            dna_start_t = t + 1
            start_t = t + 1

        if codon_start == 0 and (st == M1 or st == M4 or st == M1r or st == M4r):
            dna_len = 0
            if write:
                dna_buf[0] = raw_seq[t]
            dna_len = 1
            dna_start_t_withstop = t + 1
            dna_start_t = t + 1
            curr_indel_len = 0

            if st == M1r or st == M4r:
                if t > 2:
                    dna_start_t_withstop = t - 2
                    if write:
                        dna_buf[3 : dna_len + 3] = dna_buf[:dna_len]
                        dna_buf[0] = raw_seq[t - 3]
                        dna_buf[1] = raw_seq[t - 2]
                        dna_buf[2] = raw_seq[t - 1]
                    dna_len += 3

            prev_match = st
            codon_start = 1 if st <= M6 else -1

        elif codon_start != 0 and (st == E or st == Er or t == seq_len - 1):
            left_truncated = (vpath[start_t - 1] != S) and (vpath[start_t - 1] != Sr)
            if st == E or st == Er:
                end_t = t + 3
                right_truncated = False
            else:
                right_truncated = True
                temp_t = t
                while (
                    vpath[temp_t] != M1
                    and vpath[temp_t] != M4
                    and vpath[temp_t] != M1r
                    and vpath[temp_t] != M4r
                    and temp_t > 0
                ):
                    if dna_len > 0:
                        dna_len -= 1
                    temp_t -= 1
                end_t = temp_t

            is_partial_gene = (left_truncated and is_contig_left) or (right_truncated and is_contig_right)
            threshold = gene_len + (3 if whole_genome and codon_start == -1 else 0)

            if codon_start == 1:
                if 1 <= dna_start_t - start_t <= 3 and not left_truncated:
                    dna_start_t -= 3
                    start_t = dna_start_t
                    if write:
                        dna_buf[3 : dna_len + 3] = dna_buf[:dna_len]
                        dna_buf[0] = raw_seq[dna_start_t - 1]
                        dna_buf[1] = raw_seq[dna_start_t]
                        dna_buf[2] = raw_seq[dna_start_t + 1]
                    dna_len += 3

                if whole_genome:
                    if not left_truncated:
                        start_old = start_t
                        s = 0
                        e_save = 1e18
                        s_save = 0
                        pw = 2.5 if include_phage else 0.0
                        up_pen = 0.015 if include_phage else 0.0
                        up_ll = 0.0
                        while start_old >= 1 + s + 35:
                            pos = start_old - 1 - s
                            if (
                                include_phage
                                and s > 0
                                and prev_local_e > 0
                                and start_old - 1 >= prev_local_e - 36
                                and pos < prev_local_e - 60
                            ):
                                break
                            tri = int(tri_arr[pos])
                            if (s > 0 if include_phage else True) and stop_mask_64[tri]:
                                break
                            if include_phage and s > 0:
                                for k_f in range(3):
                                    t_c = pos + 3 + k_f
                                    if 2 <= t_c < seq_len:
                                        b0, b1, b2 = int(seq[t_c - 2]), int(seq[t_c - 1]), int(seq[t_c])
                                        if b0 <= 3 and b1 <= 3 and b2 <= 3:
                                            cg_c = int(cg_bins[t_c])
                                            up_ll += e_match[0, cg_c, k_f, b0 * 4 + b1, b2] - tr_rr[cg_c, b1, b2]
                            if start_mask_64[tri]:
                                cg_b = int(cg_bins[pos])
                                utr_start = pos - 30
                                freq_sum = 0.0
                                for j in range(61):
                                    freq_sum -= tr_start[0, cg_b, j, tri_arr[utr_start + j]]
                                score = freq_sum - pw * start_log_priors_64[tri] - up_ll + up_pen * s
                                if s == 0 or score < e_save:
                                    e_save = score
                                    s_save = s
                            elif (
                                include_phage
                                and s > 0
                                and up_ll > 7.0
                                and (tri == 30 or tri == 15 or tri == 13 or tri == 12)
                            ):
                                cg_b = int(cg_bins[pos])
                                utr_start = pos - 30
                                freq_sum = 0.0
                                for j in range(61):
                                    if j != 30:
                                        freq_sum -= tr_start[0, cg_b, j, tri_arr[utr_start + j]]
                                score = freq_sum + 12.0 - up_ll + up_pen * s
                                if score < e_save:
                                    e_save = score
                                    s_save = s
                            s += 3

                        if include_phage and curr_indel_len == 0:
                            max_d = min(180, (end_t - start_old) - gene_len)
                            d = 3
                            down_ll = 0.0
                            while d <= max_d and start_old - 1 + d + 35 < seq_len:
                                pos = start_old - 1 + d
                                for k_f in range(3):
                                    t_c = pos + k_f
                                    if 2 <= t_c < seq_len:
                                        b0, b1, b2 = int(seq[t_c - 2]), int(seq[t_c - 1]), int(seq[t_c])
                                        if b0 <= 3 and b1 <= 3 and b2 <= 3:
                                            cg_c = int(cg_bins[t_c])
                                            down_ll += e_match[0, cg_c, k_f, b0 * 4 + b1, b2] - tr_rr[cg_c, b1, b2]
                                tri = int(tri_arr[pos])
                                if stop_mask_64[tri]:
                                    break
                                if start_mask_64[tri] and down_ll < 2.5:
                                    cg_b = int(cg_bins[pos])
                                    utr_start = pos - 30
                                    freq_sum = 0.0
                                    for j in range(61):
                                        freq_sum -= tr_start[0, cg_b, j, tri_arr[utr_start + j]]
                                    score = freq_sum - pw * start_log_priors_64[tri] + 1.5 * down_ll + 0.04 * d
                                    if score < e_save:
                                        e_save = score
                                        s_save = -d
                                d += 3

                        dna_start_t -= s_save
                        if s_save > 0:
                            if write:
                                dna_buf[s_save : dna_len + s_save] = dna_buf[:dna_len]
                                for idx_s in range(s_save):
                                    dna_buf[idx_s] = raw_seq[dna_start_t - 1 + idx_s]
                            dna_len += s_save
                        elif s_save < 0:
                            d_shift = -s_save
                            dna_len -= d_shift
                            if write and dna_len > 0:
                                dna_buf[:dna_len] = dna_buf[d_shift : d_shift + dna_len]

                    if not right_truncated:
                        if write:
                            dna_buf[dna_len] = raw_seq[end_t - 3]
                            dna_buf[dna_len + 1] = raw_seq[end_t - 2]
                            dna_buf[dna_len + 2] = raw_seq[end_t - 1]
                        dna_len += 3

                if dna_len > threshold:
                    local_s = dna_start_t - 1
                    local_e = end_t
                    shadowed = False
                    if include_phage and whole_genome and not is_partial_gene and dna_len <= 300:
                        shadowed = _is_shadowed_by_opposite_orf(
                            seq,
                            tri_arr,
                            prev_local_e,
                            local_s,
                            local_e,
                            1,
                            cg_bins,
                            e_match,
                            tr_rr,
                            tr_start,
                            tr_stop,
                            dist_start,
                            dist_stop,
                            start_mask_64,
                            stop_mask_64,
                            rev_start_mask_64,
                            rev_stop_mask_64,
                            start_log_priors_64,
                            rev_start_log_priors_64,
                            stop_log_priors_64,
                            rev_stop_log_priors_64,
                        )
                    if not shadowed:
                        if include_phage and whole_genome and local_s - prev_local_e >= 60:
                            cand_s, cand_e, cand_st = _find_best_gap_orf(
                                seq,
                                tri_arr,
                                prev_local_e,
                                local_s,
                                gene_len,
                                cg_bins,
                                e_match,
                                tr_rr,
                                tr_start,
                                tr_stop,
                                dist_start,
                                dist_stop,
                                start_mask_64,
                                stop_mask_64,
                                rev_start_mask_64,
                                rev_stop_mask_64,
                                start_log_priors_64,
                                rev_start_log_priors_64,
                                stop_log_priors_64,
                                rev_stop_log_priors_64,
                            )
                            if cand_st != 0:
                                cg_s = c_start + cand_s
                                cg_e = c_start + cand_e
                                cg_mid = (cg_s + cg_e) // 2
                                cand_len = cand_e - cand_s
                                if left_cutoff <= cg_mid < right_cutoff and cg_e > cg_s and cand_len >= 60:
                                    if write:
                                        g_slot = gene_off + n_genes
                                        c_slot = cds_off + total_cds
                                        i_slot = indel_off + total_indels
                                        out_starts[g_slot] = cg_s
                                        out_ends[g_slot] = cg_e
                                        out_strands[g_slot] = cand_st
                                        out_partials[g_slot] = False
                                        out_contig_indices[g_slot] = contig_idx
                                        out_cds_offsets[g_slot] = c_slot
                                        out_cds_lengths[g_slot] = cand_len
                                        if cand_st == 1:
                                            out_cds_seqs[c_slot : c_slot + cand_len] = raw_seq[cand_s:cand_e]
                                        else:
                                            for k in range(cand_len):
                                                out_cds_seqs[c_slot + k] = lut_rc[raw_seq[cand_e - 1 - k]]
                                        out_indel_offsets[g_slot] = i_slot
                                        out_indel_lengths[g_slot] = 0
                                    n_genes += 1
                                    total_cds += cand_len

                        g_s = c_start + local_s
                        g_e = c_start + local_e
                        mid = (g_s + g_e) // 2
                        if left_cutoff <= mid < right_cutoff and g_e > g_s and dna_len >= 60:
                            if write:
                                g_slot = gene_off + n_genes
                                c_slot = cds_off + total_cds
                                i_slot = indel_off + total_indels
                                out_starts[g_slot] = g_s
                                out_ends[g_slot] = g_e
                                out_strands[g_slot] = 1
                                out_partials[g_slot] = is_partial_gene
                                out_contig_indices[g_slot] = contig_idx
                                out_cds_offsets[g_slot] = c_slot
                                out_cds_lengths[g_slot] = dna_len
                                out_cds_seqs[c_slot : c_slot + dna_len] = dna_buf[:dna_len]
                                out_indel_offsets[g_slot] = i_slot
                                out_indel_lengths[g_slot] = curr_indel_len
                                for ii in range(curr_indel_len):
                                    out_indel_pos[i_slot + ii] = indel_pos_buf[ii] + c_start
                                    out_indel_kinds[i_slot + ii] = indel_kind_buf[ii]
                            n_genes += 1
                            total_cds += dna_len
                            total_indels += curr_indel_len
                        if local_e > prev_local_e:
                            prev_local_e = local_e

            elif codon_start == -1:
                if whole_genome and not right_truncated:
                    end_old = end_t
                    s = 0
                    e_save = 1e18
                    s_save = 0
                    pw = 2.5 if include_phage else 0.0
                    up_pen = 0.015 if include_phage else 0.0
                    up_ll = 0.0
                    while end_old - 2 + s + 35 < seq_len:
                        pos = end_old - 3 + s
                        tri = int(tri_arr[pos])
                        if (s > 0 if include_phage else True) and rev_stop_mask_64[tri]:
                            break
                        if include_phage and s > 0:
                            for k_f in range(3):
                                t_c = pos - 3 + k_f
                                if 2 <= t_c < seq_len:
                                    b0, b1, b2 = int(seq[t_c - 2]), int(seq[t_c - 1]), int(seq[t_c])
                                    if b0 <= 3 and b1 <= 3 and b2 <= 3:
                                        cg_c = int(cg_bins[t_c])
                                        up_ll += e_match[1, cg_c, k_f, b0 * 4 + b1, b2] - tr_rr[cg_c, b1, b2]
                        if rev_start_mask_64[tri]:
                            cg_b = int(cg_bins[pos])
                            utr_start = pos - 30
                            freq_sum = 0.0
                            for j in range(61):
                                freq_sum -= tr_stop[1, cg_b, j, tri_arr[utr_start + j]]
                            score = freq_sum - pw * rev_start_log_priors_64[tri] - up_ll + up_pen * s
                            if s == 0 or score < e_save:
                                e_save = score
                                s_save = s
                        elif (
                            include_phage
                            and s > 0
                            and up_ll > 7.0
                            and (tri == 18 or tri == 3 or tri == 35 or tri == 51)
                        ):
                            cg_b = int(cg_bins[pos])
                            utr_start = pos - 30
                            freq_sum = 0.0
                            for j in range(61):
                                if j != 30:
                                    freq_sum -= tr_stop[1, cg_b, j, tri_arr[utr_start + j]]
                            score = freq_sum + 12.0 - up_ll + up_pen * s
                            if score < e_save:
                                e_save = score
                                s_save = s
                        s += 3

                    if include_phage and curr_indel_len == 0:
                        max_d = min(180, (end_old - dna_start_t_withstop) - gene_len)
                        d = 3
                        down_ll = 0.0
                        while d <= max_d and end_old - 3 - d >= 35:
                            pos = end_old - 3 - d
                            for k_f in range(3):
                                t_c = pos + k_f
                                if 2 <= t_c < seq_len:
                                    b0, b1, b2 = int(seq[t_c - 2]), int(seq[t_c - 1]), int(seq[t_c])
                                    if b0 <= 3 and b1 <= 3 and b2 <= 3:
                                        cg_c = int(cg_bins[t_c])
                                        down_ll += e_match[1, cg_c, k_f, b0 * 4 + b1, b2] - tr_rr[cg_c, b1, b2]
                            tri = int(tri_arr[pos])
                            if rev_stop_mask_64[tri]:
                                break
                            if rev_start_mask_64[tri] and down_ll < 2.5:
                                cg_b = int(cg_bins[pos])
                                utr_start = pos - 30
                                freq_sum = 0.0
                                for j in range(61):
                                    freq_sum -= tr_stop[1, cg_b, j, tri_arr[utr_start + j]]
                                score = freq_sum - pw * rev_start_log_priors_64[tri] + 1.5 * down_ll + 0.04 * d
                                if score < e_save:
                                    e_save = score
                                    s_save = -d
                            d += 3

                    end_t = end_old + s_save
                    if s_save >= 0:
                        if write:
                            for idx_s in range(s_save + 3):
                                dna_buf[dna_len + idx_s] = raw_seq[end_old - 3 + idx_s]
                        dna_len += s_save + 3
                    else:
                        dna_len += s_save
                        if write:
                            dna_buf[dna_len] = raw_seq[end_t - 3]
                            dna_buf[dna_len + 1] = raw_seq[end_t - 2]
                            dna_buf[dna_len + 2] = raw_seq[end_t - 1]
                        dna_len += 3

                if dna_len > threshold:
                    local_s = dna_start_t_withstop - 1
                    local_e = end_t
                    shadowed = False
                    if include_phage and whole_genome and not is_partial_gene and dna_len <= 300:
                        shadowed = _is_shadowed_by_opposite_orf(
                            seq,
                            tri_arr,
                            prev_local_e,
                            local_s,
                            local_e,
                            -1,
                            cg_bins,
                            e_match,
                            tr_rr,
                            tr_start,
                            tr_stop,
                            dist_start,
                            dist_stop,
                            start_mask_64,
                            stop_mask_64,
                            rev_start_mask_64,
                            rev_stop_mask_64,
                            start_log_priors_64,
                            rev_start_log_priors_64,
                            stop_log_priors_64,
                            rev_stop_log_priors_64,
                        )
                    if not shadowed:
                        if include_phage and whole_genome and local_s - prev_local_e >= 60:
                            cand_s, cand_e, cand_st = _find_best_gap_orf(
                                seq,
                                tri_arr,
                                prev_local_e,
                                local_s,
                                gene_len,
                                cg_bins,
                                e_match,
                                tr_rr,
                                tr_start,
                                tr_stop,
                                dist_start,
                                dist_stop,
                                start_mask_64,
                                stop_mask_64,
                                rev_start_mask_64,
                                rev_stop_mask_64,
                                start_log_priors_64,
                                rev_start_log_priors_64,
                                stop_log_priors_64,
                                rev_stop_log_priors_64,
                            )
                            if cand_st != 0:
                                cg_s = c_start + cand_s
                                cg_e = c_start + cand_e
                                cg_mid = (cg_s + cg_e) // 2
                                cand_len = cand_e - cand_s
                                if left_cutoff <= cg_mid < right_cutoff and cg_e > cg_s and cand_len >= 60:
                                    if write:
                                        g_slot = gene_off + n_genes
                                        c_slot = cds_off + total_cds
                                        i_slot = indel_off + total_indels
                                        out_starts[g_slot] = cg_s
                                        out_ends[g_slot] = cg_e
                                        out_strands[g_slot] = cand_st
                                        out_partials[g_slot] = False
                                        out_contig_indices[g_slot] = contig_idx
                                        out_cds_offsets[g_slot] = c_slot
                                        out_cds_lengths[g_slot] = cand_len
                                        if cand_st == 1:
                                            out_cds_seqs[c_slot : c_slot + cand_len] = raw_seq[cand_s:cand_e]
                                        else:
                                            for k in range(cand_len):
                                                out_cds_seqs[c_slot + k] = lut_rc[raw_seq[cand_e - 1 - k]]
                                        out_indel_offsets[g_slot] = i_slot
                                        out_indel_lengths[g_slot] = 0
                                    n_genes += 1
                                    total_cds += cand_len

                        g_s = c_start + local_s
                        g_e = c_start + local_e
                        mid = (g_s + g_e) // 2
                        if left_cutoff <= mid < right_cutoff and g_e > g_s and dna_len >= 60:
                            if write:
                                g_slot = gene_off + n_genes
                                c_slot = cds_off + total_cds
                                i_slot = indel_off + total_indels
                                out_starts[g_slot] = g_s
                                out_ends[g_slot] = g_e
                                out_strands[g_slot] = -1
                                out_partials[g_slot] = is_partial_gene
                                out_contig_indices[g_slot] = contig_idx
                                out_cds_offsets[g_slot] = c_slot
                                out_cds_lengths[g_slot] = dna_len
                                for k in range(dna_len):
                                    out_cds_seqs[c_slot + k] = lut_rc[dna_buf[dna_len - 1 - k]]
                                out_indel_offsets[g_slot] = i_slot
                                out_indel_lengths[g_slot] = curr_indel_len
                                for ii in range(curr_indel_len):
                                    out_indel_pos[i_slot + ii] = indel_pos_buf[ii] + c_start
                                    out_indel_kinds[i_slot + ii] = indel_kind_buf[ii]
                            n_genes += 1
                            total_cds += dna_len
                            total_indels += curr_indel_len
                        if local_e > prev_local_e:
                            prev_local_e = local_e

            codon_start = 0
            start_t = -1
            dna_len = 0
            curr_indel_len = 0

        elif codon_start != 0 and (
            ((M1 <= st <= M6) and prev_match >= M1) or ((M1r <= st <= M6r) and prev_match >= M1r)
        ):
            if st < prev_match:
                out_nt = st + 6 - prev_match
            else:
                out_nt = st - prev_match
            for kk in range(out_nt):
                if write:
                    dna_buf[dna_len] = ord("N")
                dna_len += 1
                if kk > 0:
                    if write:
                        indel_pos_buf[curr_indel_len] = t
                        indel_kind_buf[curr_indel_len] = -1
                    curr_indel_len += 1
            if dna_len > 0:
                dna_len -= 1
            if write:
                dna_buf[dna_len] = raw_seq[t]
            dna_len += 1
            prev_match = st

        elif codon_start != 0 and ((I1 <= st <= I6) or (I1r <= st <= I6r)):
            if write:
                indel_pos_buf[curr_indel_len] = t
                indel_kind_buf[curr_indel_len] = 1
            curr_indel_len += 1

        elif codon_start != 0 and st == R:
            codon_start = 0
            start_t = -1
            dna_len = 0
            curr_indel_len = 0

    return n_genes, total_cds, total_indels


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _decode_and_count_windows(
    flat_seqs: np.ndarray,  # type: ignore[type-arg]
    encode_lut: np.ndarray,  # type: ignore[type-arg]
    lut_rc: np.ndarray,  # type: ignore[type-arg]
    win_contig_idx: np.ndarray,  # type: ignore[type-arg]
    win_buf_starts: np.ndarray,  # type: ignore[type-arg]
    win_buf_ends: np.ndarray,  # type: ignore[type-arg]
    win_c_starts: np.ndarray,  # type: ignore[type-arg]
    win_left_cutoffs: np.ndarray,  # type: ignore[type-arg]
    win_right_cutoffs: np.ndarray,  # type: ignore[type-arg]
    win_is_left_edge: np.ndarray,  # type: ignore[type-arg]
    win_is_right_edge: np.ndarray,  # type: ignore[type-arg]
    win_vp_offsets: np.ndarray,  # type: ignore[type-arg]
    whole_genome: bool,
    include_phage: bool,
    min_gene_len: int,
    local_gc_window: int,
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
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint8]],
    np.ndarray[tuple[int], np.dtype[np.int8]],
    np.ndarray[tuple[int], np.dtype[np.int8]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    r"""Parallel pass 1: encode windows, run 29-state Viterbi decoding, and count gene/CDS/indel sizes."""
    n_windows = len(win_contig_idx)
    total_vp_len = int(win_vp_offsets[n_windows])

    encoded_flat = np.empty(total_vp_len, dtype=np.uint8)
    vpath_flat = np.empty(total_vp_len, dtype=np.int8)
    cg_bins_flat = np.empty(total_vp_len, dtype=np.int8)
    win_gene_counts = np.empty(n_windows, dtype=np.int32)
    win_cds_bytes = np.empty(n_windows, dtype=np.int32)
    win_indel_counts = np.empty(n_windows, dtype=np.int32)

    dummy_i32 = np.empty(0, dtype=np.int32)
    dummy_i8 = np.empty(0, dtype=np.int8)
    dummy_u8 = np.empty(0, dtype=np.uint8)
    dummy_u32 = np.empty(0, dtype=np.uint32)
    dummy_bool = np.empty(0, dtype=np.bool_)

    for w in prange(n_windows):  # type: ignore[not-iterable]
        b_start = int(win_buf_starts[w])
        b_end = int(win_buf_ends[w])
        w_len = b_end - b_start
        vp_off = int(win_vp_offsets[w])

        raw_sub = flat_seqs[b_start:b_end]
        enc_sub = encoded_flat[vp_off : vp_off + w_len]

        c_count = 0
        for i in range(w_len):
            code = encode_lut[raw_sub[i]]
            enc_sub[i] = code
            if code == 1 or code == 2:
                c_count += 1

        cg_pct = (c_count * 100) // w_len
        cg_bin = min(43, max(0, cg_pct - 26))
        cg_sub = _build_local_cg_bins(enc_sub, cg_bin, local_gc_window if include_phage else 0)
        cg_bins_flat[vp_off : vp_off + w_len] = cg_sub

        best_st, path = _viterbi_forward(
            enc_sub,
            cg_sub,
            whole_genome,
            include_phage,
            tr,
            tr_mi,
            tr_ii,
            pi,
            tr_rr,
            e_match,
            tr_start,
            tr_stop,
            dist_start,
            dist_stop,
            start_mask_64,
            stop_mask_64,
            rev_start_mask_64,
            rev_stop_mask_64,
            start_log_priors_64,
            rev_start_log_priors_64,
            stop_log_priors_64,
            rev_stop_log_priors_64,
        )
        vp_sub = _viterbi_backtrack(best_st, path)
        vpath_flat[vp_off : vp_off + w_len] = vp_sub

        g_cnt, c_bytes, i_cnt = _scan_window_genes(
            raw_sub,
            enc_sub,
            vp_sub,
            cg_sub,
            whole_genome,
            include_phage,
            min_gene_len,
            e_match,
            tr_rr,
            tr_start,
            tr_stop,
            dist_start,
            dist_stop,
            start_mask_64,
            stop_mask_64,
            rev_start_mask_64,
            rev_stop_mask_64,
            start_log_priors_64,
            rev_start_log_priors_64,
            stop_log_priors_64,
            rev_stop_log_priors_64,
            lut_rc,
            int(win_contig_idx[w]),
            int(win_c_starts[w]),
            int(win_left_cutoffs[w]),
            int(win_right_cutoffs[w]),
            bool(win_is_left_edge[w]),
            bool(win_is_right_edge[w]),
            False,
            dummy_i32,
            dummy_i32,
            dummy_i8,
            dummy_bool,
            dummy_u32,
            dummy_u8,
            dummy_i32,
            dummy_i32,
            dummy_i32,
            dummy_i8,
            dummy_i32,
            dummy_i32,
            0,
            0,
            0,
        )
        win_gene_counts[w] = g_cnt
        win_cds_bytes[w] = c_bytes
        win_indel_counts[w] = i_cnt

    return encoded_flat, vpath_flat, cg_bins_flat, win_gene_counts, win_cds_bytes, win_indel_counts


@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def _write_windows_parallel(
    flat_seqs: np.ndarray,  # type: ignore[type-arg]
    encoded_flat: np.ndarray,  # type: ignore[type-arg]
    vpath_flat: np.ndarray,  # type: ignore[type-arg]
    cg_bins_flat: np.ndarray,  # type: ignore[type-arg]
    lut_rc: np.ndarray,  # type: ignore[type-arg]
    win_contig_idx: np.ndarray,  # type: ignore[type-arg]
    win_buf_starts: np.ndarray,  # type: ignore[type-arg]
    win_buf_ends: np.ndarray,  # type: ignore[type-arg]
    win_c_starts: np.ndarray,  # type: ignore[type-arg]
    win_left_cutoffs: np.ndarray,  # type: ignore[type-arg]
    win_right_cutoffs: np.ndarray,  # type: ignore[type-arg]
    win_is_left_edge: np.ndarray,  # type: ignore[type-arg]
    win_is_right_edge: np.ndarray,  # type: ignore[type-arg]
    win_vp_offsets: np.ndarray,  # type: ignore[type-arg]
    win_gene_offsets: np.ndarray,  # type: ignore[type-arg]
    win_gene_counts: np.ndarray,  # type: ignore[type-arg]
    win_cds_offsets: np.ndarray,  # type: ignore[type-arg]
    win_indel_offsets: np.ndarray,  # type: ignore[type-arg]
    whole_genome: bool,
    include_phage: bool,
    min_gene_len: int,
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
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
    out_starts: np.ndarray,  # type: ignore[type-arg]
    out_ends: np.ndarray,  # type: ignore[type-arg]
    out_strands: np.ndarray,  # type: ignore[type-arg]
    out_partials: np.ndarray,  # type: ignore[type-arg]
    out_contig_indices: np.ndarray,  # type: ignore[type-arg]
    out_cds_seqs: np.ndarray,  # type: ignore[type-arg]
    out_cds_offsets: np.ndarray,  # type: ignore[type-arg]
    out_cds_lengths: np.ndarray,  # type: ignore[type-arg]
    out_indel_pos: np.ndarray,  # type: ignore[type-arg]
    out_indel_kinds: np.ndarray,  # type: ignore[type-arg]
    out_indel_offsets: np.ndarray,  # type: ignore[type-arg]
    out_indel_lengths: np.ndarray,  # type: ignore[type-arg]
) -> None:
    r"""Parallel pass 2: populate pre-allocated SoA gene arrays directly across all decoded windows."""
    n_windows = len(win_contig_idx)
    for w in prange(n_windows):  # type: ignore[not-iterable]
        if win_gene_counts[w] == 0:
            continue
        b_start = int(win_buf_starts[w])
        b_end = int(win_buf_ends[w])
        w_len = b_end - b_start
        vp_off = int(win_vp_offsets[w])

        _scan_window_genes(
            flat_seqs[b_start:b_end],
            encoded_flat[vp_off : vp_off + w_len],
            vpath_flat[vp_off : vp_off + w_len],
            cg_bins_flat[vp_off : vp_off + w_len],
            whole_genome,
            include_phage,
            min_gene_len,
            e_match,
            tr_rr,
            tr_start,
            tr_stop,
            dist_start,
            dist_stop,
            start_mask_64,
            stop_mask_64,
            rev_start_mask_64,
            rev_stop_mask_64,
            start_log_priors_64,
            rev_start_log_priors_64,
            stop_log_priors_64,
            rev_stop_log_priors_64,
            lut_rc,
            int(win_contig_idx[w]),
            int(win_c_starts[w]),
            int(win_left_cutoffs[w]),
            int(win_right_cutoffs[w]),
            bool(win_is_left_edge[w]),
            bool(win_is_right_edge[w]),
            True,
            out_starts,
            out_ends,
            out_strands,
            out_partials,
            out_contig_indices,
            out_cds_seqs,
            out_cds_offsets,
            out_cds_lengths,
            out_indel_pos,
            out_indel_kinds,
            out_indel_offsets,
            out_indel_lengths,
            int(win_gene_offsets[w]),
            int(win_cds_offsets[w]),
            int(win_indel_offsets[w]),
        )


@njit(fastmath=True, nogil=True, cache=True)
def _extract_genes(
    raw_seq: np.ndarray,  # type: ignore[type-arg]
    seq: np.ndarray,  # type: ignore[type-arg]
    vpath: np.ndarray,  # type: ignore[type-arg]
    cg_bins: np.ndarray,  # type: ignore[type-arg]
    whole_genome: bool,
    include_phage: bool,
    min_gene_len: int,
    e_match: np.ndarray,  # type: ignore[type-arg]
    tr_rr: np.ndarray,  # type: ignore[type-arg]
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
    lut_rc: np.ndarray,  # type: ignore[type-arg]
) -> tuple[list[int], list[int], list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:  # type: ignore[type-arg]
    r"""Extract genes from a single chunk into lists (legacy compatibility helper)."""
    seq_len = len(seq)
    dummy_i32 = np.empty(0, dtype=np.int32)
    dummy_i8 = np.empty(0, dtype=np.int8)
    dummy_u8 = np.empty(0, dtype=np.uint8)
    dummy_u32 = np.empty(0, dtype=np.uint32)
    dummy_bool = np.empty(0, dtype=np.bool_)

    n_genes, total_cds, total_indels = _scan_window_genes(
        raw_seq,
        seq,
        vpath,
        cg_bins,
        whole_genome,
        include_phage,
        min_gene_len,
        e_match,
        tr_rr,
        tr_start,
        tr_stop,
        dist_start,
        dist_stop,
        start_mask_64,
        stop_mask_64,
        rev_start_mask_64,
        rev_stop_mask_64,
        start_log_priors_64,
        rev_start_log_priors_64,
        stop_log_priors_64,
        rev_stop_log_priors_64,
        lut_rc,
        0,
        0,
        0,
        seq_len,
        True,
        True,
        False,
        dummy_i32,
        dummy_i32,
        dummy_i8,
        dummy_bool,
        dummy_u32,
        dummy_u8,
        dummy_i32,
        dummy_i32,
        dummy_i32,
        dummy_i8,
        dummy_i32,
        dummy_i32,
        0,
        0,
        0,
    )

    starts_arr = np.empty(n_genes, dtype=np.int32)
    ends_arr = np.empty(n_genes, dtype=np.int32)
    strands_arr = np.empty(n_genes, dtype=np.int8)
    partials_arr = np.empty(n_genes, dtype=np.bool_)
    contig_arr = np.empty(n_genes, dtype=np.uint32)
    cds_seqs = np.empty(total_cds, dtype=np.uint8)
    cds_offsets = np.empty(n_genes, dtype=np.int32)
    cds_lengths = np.empty(n_genes, dtype=np.int32)
    indel_pos = np.empty(total_indels, dtype=np.int32)
    indel_kinds = np.empty(total_indels, dtype=np.int8)
    indel_offsets = np.empty(n_genes, dtype=np.int32)
    indel_lengths = np.empty(n_genes, dtype=np.int32)

    if n_genes > 0:
        _scan_window_genes(
            raw_seq,
            seq,
            vpath,
            cg_bins,
            whole_genome,
            include_phage,
            min_gene_len,
            e_match,
            tr_rr,
            tr_start,
            tr_stop,
            dist_start,
            dist_stop,
            start_mask_64,
            stop_mask_64,
            rev_start_mask_64,
            rev_stop_mask_64,
            start_log_priors_64,
            rev_start_log_priors_64,
            stop_log_priors_64,
            rev_stop_log_priors_64,
            lut_rc,
            0,
            0,
            0,
            seq_len,
            True,
            True,
            True,
            starts_arr,
            ends_arr,
            strands_arr,
            partials_arr,
            contig_arr,
            cds_seqs,
            cds_offsets,
            cds_lengths,
            indel_pos,
            indel_kinds,
            indel_offsets,
            indel_lengths,
            0,
            0,
            0,
        )

    starts: list[int] = [int(starts_arr[i]) for i in range(n_genes)]
    ends: list[int] = [int(ends_arr[i]) for i in range(n_genes)]
    strands: list[int] = [int(strands_arr[i]) for i in range(n_genes)]
    cds_list: list[np.ndarray] = [  # type: ignore[type-arg]
        cds_seqs[cds_offsets[i] : cds_offsets[i] + cds_lengths[i]].copy() for i in range(n_genes)
    ]
    indel_pos_list: list[np.ndarray] = [  # type: ignore[type-arg]
        indel_pos[indel_offsets[i] : indel_offsets[i] + indel_lengths[i]].copy() for i in range(n_genes)
    ]
    indel_kind_list: list[np.ndarray] = [  # type: ignore[type-arg]
        indel_kinds[indel_offsets[i] : indel_offsets[i] + indel_lengths[i]].copy() for i in range(n_genes)
    ]
    return starts, ends, strands, cds_list, indel_pos_list, indel_kind_list
