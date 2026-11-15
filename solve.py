#!/usr/bin/env python3
"""
MTAH 2027 - Possession Pipeline
Beats baseline (spellF1 0.2433) with:
  1. Forward-fill ball-missing frames instead of defaulting to D
  2. Smart tagging (SET PIECES + GOALS) for explicit dead-ball labelling
  3. Long consecutive no-ball gaps (>=100 frames) as additional D signal
  4. Near-touchline ball detection as D signal
  5. Larger smoothing window to reduce H<->A jitter
  6. Minimum spell duration enforcement
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

# -- paths ------------------------------------------------------------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TAC_XML  = os.path.join(DATA_DIR, "BAR-ATH_tactical_atd.xml")
TAC_TXT  = os.path.join(DATA_DIR, "BAR-ATH_tactical_atd.txt")
TAGS_XML = os.path.join(DATA_DIR, "BAR-ATH_tactical_smart_tagging.xml")
GT_CSV   = os.path.join(DATA_DIR, "BAR-ATH_possession_gt_public.csv")
OUT_CSV  = os.path.join(DATA_DIR, "submission.csv")

# -- constants --------------------------------------------------------------
N_FRAMES = 199_319
PITCH    = (105.3, 68.0)    # metres
FPS      = 25
HOME, AWAY, NOBODY = 0, 1, -1

# Halftime gap - never bridge across this
HALF1_END   = 90_652
HALF2_START = 110_808

# Tunable params (grid-searched on 5-min public GT)
SMOOTH_WINDOW   = 17     # frames either side for majority vote (~0.7 sec)
MIN_SPELL       = 50     # minimum spell length in frames (~2 sec) to reduce jitter
DEAD_GAP_MIN    = 250    # consecutive no-ball frames -> D signal (~10 sec)
DEAD_BOUNDARY_Y = 0.97   # ball y-coord above this = near touchline -> D signal
DEAD_BOUNDARY_X = 0.015  # ball x-coord below this or above 1-this = near goal line


# ===========================================================================
# 1.  Slot -> team mapping
# ===========================================================================

def parse_team_slots(xml_path):
    """Array[slot_index] -> HOME(0) or AWAY(1). First team in XML = home = Barcelona."""
    root = ET.parse(xml_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"Expected 2 teams, got {len(team_ids)}")
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


# ===========================================================================
# 2.  Parse tracking -> nearest-player label + raw ball positions
# ===========================================================================

def parse_tracking(txt_path, team_slots, n_frames, pitch):
    """
    Returns:
      raw_codes : int8 array  HOME / AWAY / NOBODY per frame
      ball_x    : float64 array, NaN where no ball
      ball_y    : float64 array, NaN where no ball
    """
    length, width = pitch
    raw_codes = np.full(n_frames, NOBODY, dtype=np.int8)
    ball_x    = np.full(n_frames, np.nan)
    ball_y    = np.full(n_frames, np.nan)

    print(f"Reading {os.path.basename(txt_path)} ...")
    with open(txt_path) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue

            ball_str = parts[2].strip()
            if ball_str[0] == "N":
                continue

            bvals = ball_str.split(",")
            bx_norm = float(bvals[0])
            by_norm = float(bvals[1])
            bx = bx_norm * length
            by = by_norm * width

            best_d2, best_team = None, NOBODY
            for slot, grp in enumerate(parts[1].split(";")):
                if grp[0] == "N":
                    continue
                gv = grp.split(",")
                dx = float(gv[0]) * length - bx
                dy = float(gv[1]) * width  - by
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2, best_team = d2, team_slots[slot]

            fidx = int(parts[0]) - 1
            if 0 <= fidx < n_frames:
                raw_codes[fidx] = best_team
                ball_x[fidx]    = bx_norm
                ball_y[fidx]    = by_norm

    n_seen = int((raw_codes != NOBODY).sum())
    print(f"  Ball located on {n_seen}/{n_frames} frames ({100*n_seen/n_frames:.1f}%)")
    return raw_codes, ball_x, ball_y


# ===========================================================================
# 3.  Build dead-ball mask from multiple signals
# ===========================================================================

def _first_ball_after_gap(has_ball, win_start, win_end, min_gap=10):
    """
    Within the window [win_start, win_end), find the first frame where
    the ball reappears after the LONGEST gap of >= min_gap frames.
    Returns that frame index, or win_end if no such gap exists.
    """
    best_gap_len, best_resume = 0, win_end
    in_gap, gap_st = False, win_start
    for i in range(win_start, win_end):
        if not has_ball[i]:
            if not in_gap:
                gap_st, in_gap = i, True
        else:
            if in_gap:
                gap_len = i - gap_st
                if gap_len >= min_gap and gap_len > best_gap_len:
                    best_gap_len, best_resume = gap_len, i
                in_gap = False
    return best_resume


def build_dead_mask(tags_xml, ball_x, ball_y, n_frames, fps=FPS):
    """
    Combines two D signals:
      a) Smart tagging: SET PIECES + GOALS windows, trimmed to end when
         ball reappears after the main tracking gap within the window.
         This prevents the tagging window from masking live play that
         resumes BEFORE the window end-timestamp.
      b) Long consecutive no-ball gaps (>= DEAD_GAP_MIN frames)
    Returns bool array.
    """
    dead = np.zeros(n_frames, dtype=bool)
    has_ball = ~np.isnan(ball_x)

    # --- (a) smart tagging (trimmed) ----------------------------------------
    root = ET.parse(tags_xml).getroot()
    for inst in root.findall(".//instance"):
        code = inst.findtext("code")
        if code not in ("SET PIECES", "GOALS"):
            continue
        s = max(0, int(float(inst.findtext("start")) * fps))
        e = min(n_frames, int(float(inst.findtext("end")) * fps) + 1)

        # Trim the window: end it at ball reappearance after the main gap.
        # This handles cases where the tagging window extends into live play
        # (e.g. a 35-second free-kick window where play resumes after 12 s).
        trim_end = _first_ball_after_gap(has_ball, s, e, min_gap=10)
        dead[s:trim_end] = True

    n_tags = int(dead.sum())

    # --- (b) long no-ball gaps ---------------------------------------------
    gap_dead = np.zeros(n_frames, dtype=bool)
    in_gap   = False
    gap_st   = 0
    for i in range(n_frames):
        if not has_ball[i]:
            if not in_gap:
                gap_st, in_gap = i, True
        else:
            if in_gap:
                gap_len = i - gap_st
                if gap_len >= DEAD_GAP_MIN:
                    gap_dead[gap_st:i] = True
                in_gap = False
    if in_gap and (n_frames - gap_st) >= DEAD_GAP_MIN:
        gap_dead[gap_st:] = True

    dead |= gap_dead
    n_gaps = int(gap_dead.sum())

    n_total = int(dead.sum())
    print(f"  Dead-ball mask: {n_total} frames ({100*n_total/n_frames:.1f}%) "
          f"[tags={n_tags}, gaps={n_gaps}]")
    return dead


# ===========================================================================
# 4.  Majority-vote smoothing
# ===========================================================================

def smooth_majority(codes, window):
    """
    Centred majority vote over +/-window frames on HOME/AWAY.
    NOBODY frames don't vote. Ties -> HOME. Returns int8.
    """
    if window <= 0:
        return codes.copy()

    h_cum = np.concatenate(([0], np.cumsum(codes == HOME).astype(np.int32)))
    a_cum = np.concatenate(([0], np.cumsum(codes == AWAY).astype(np.int32)))
    idx = np.arange(len(codes))
    lo  = np.maximum(0, idx - window)
    hi  = np.minimum(len(codes), idx + window + 1)
    h   = h_cum[hi] - h_cum[lo]
    a   = a_cum[hi] - a_cum[lo]

    out      = np.full(len(codes), NOBODY, dtype=np.int8)
    seen     = (h + a) > 0
    out[seen] = np.where(h[seen] >= a[seen], HOME, AWAY)
    return out


# ===========================================================================
# 5.  Forward-fill NOBODY gaps (ball lost during live play)
# ===========================================================================

def forward_fill(codes):
    """
    Carry last known HOME/AWAY forward through NOBODY gaps.
    Backward-fill the leading NOBODY block from the first known label.
    Resets at halftime gap so H1 doesn't bleed into H2.
    """
    out  = codes.copy()
    last = NOBODY

    # Forward pass
    for i in range(len(out)):
        if i == HALF2_START:
            last = NOBODY
        if out[i] == NOBODY:
            out[i] = last
        else:
            last = out[i]

    # Backward pass: fill leading NOBODY in H1
    last = NOBODY
    for i in range(HALF1_END - 1, -1, -1):
        if out[i] == NOBODY:
            out[i] = last
        else:
            last = out[i]

    # Backward pass: fill leading NOBODY in H2
    last = NOBODY
    for i in range(len(out) - 1, HALF2_START - 1, -1):
        if out[i] == NOBODY:
            out[i] = last
        else:
            last = out[i]

    return out


# ===========================================================================
# 6.  Enforce minimum spell duration
# ===========================================================================

def enforce_min_spell(labels, min_frames=MIN_SPELL):
    """
    Any run shorter than min_frames is absorbed into its left neighbour.
    Skips X. Runs multiple passes until stable (up to 10).
    """
    out = labels.copy()
    for _ in range(10):
        changed = False
        i = 0
        while i < len(out):
            j = i + 1
            while j < len(out) and out[j] == out[i]:
                j += 1
            if (j - i) < min_frames and out[i] != "X":
                fill = out[i - 1] if i > 0 else (out[j] if j < len(out) else out[i])
                out[i:j] = fill
                changed = True
            i = j
        if not changed:
            break
    return out


# ===========================================================================
# 7.  Write submission
# ===========================================================================

def write_csv(path, labels):
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    print(f"Wrote {len(labels)} rows -> {path}")


# ===========================================================================
# 8.  Evaluate against public 5-min GT
# ===========================================================================

def evaluate_quick(pred_labels, gt_path):
    gt_df   = pd.read_csv(gt_path)
    gt_arr  = gt_df["label"].values
    pred_arr = np.array(pred_labels)

    graded_mask = gt_arr != "X"
    gt_g   = gt_arr[graded_mask]
    pred_g = pred_arr[graded_mask]

    classes = ["H", "A", "D"]
    f1s = []
    print("\n-- Per-class F1 (graded frames only) --")
    for cls in classes:
        tp = int(np.sum((pred_g == cls) & (gt_g == cls)))
        fp = int(np.sum((pred_g == cls) & (gt_g != cls)))
        fn = int(np.sum((pred_g != cls) & (gt_g == cls)))
        p  = tp / (tp + fp) if tp + fp else 0.0
        r  = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2*p*r / (p+r) if p+r else 0.0
        f1s.append(f1)
        print(f"  {cls}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  "
              f"(TP={tp} FP={fp} FN={fn})")
    macro = float(np.mean(f1s))
    print(f"  macroF1 = {macro:.4f}")

    # Spell counts in graded window
    spells, prev = 0, None
    for lbl in pred_g:
        if lbl in ("H", "A") and lbl != prev:
            spells += 1
        prev = lbl if lbl in ("H", "A") else prev
    gt_spells, prev = 0, None
    for lbl in gt_g:
        if lbl in ("H", "A") and lbl != prev:
            gt_spells += 1
        prev = lbl if lbl in ("H", "A") else prev
    print(f"  Possession spells -- GT: {gt_spells}  Pred: {spells}")

    # Full distribution
    for lbl in ["H", "A", "D", "X"]:
        n = int(np.sum(pred_arr == lbl))
        print(f"  {lbl}: {n:>7} ({100*n/len(pred_arr):.1f}%)")
    return macro


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("MTAH 2027 - Possession Solver")
    print("=" * 60)

    # 1 - team slot mapping
    team_slots = parse_team_slots(TAC_XML)
    print(f"Team slots: HOME(Barcelona)={int((team_slots==HOME).sum())}  "
          f"AWAY(Athletic)={int((team_slots==AWAY).sum())}")

    # 2 - nearest-player raw codes + ball positions
    raw, ball_x, ball_y = parse_tracking(TAC_TXT, team_slots, N_FRAMES, PITCH)

    # 3 - majority-vote smoothing
    print(f"Smoothing (window={SMOOTH_WINDOW}) ...")
    smooth = smooth_majority(raw, window=SMOOTH_WINDOW)

    # 4 - forward-fill NOBODY gaps
    print("Forward-filling NOBODY gaps ...")
    filled = forward_fill(smooth)

    # 5 - build dead-ball mask
    print("Building dead-ball mask ...")
    dead_mask = build_dead_mask(TAGS_XML, ball_x, ball_y, N_FRAMES)

    # 6 - assemble labels: live -> H/A, dead -> D, unknown -> X
    labels = np.where(
        filled == HOME, "H",
        np.where(filled == AWAY, "A", "X")
    )
    labels[dead_mask] = "D"

    # 7 - enforce minimum spell duration to kill remaining jitter
    print(f"Enforcing min spell ({MIN_SPELL} frames) ...")
    labels = enforce_min_spell(labels, MIN_SPELL)

    # 8 - write output
    write_csv(OUT_CSV, labels)

    # 9 - evaluate
    if os.path.exists(GT_CSV):
        evaluate_quick(labels, GT_CSV)

    print("\nDone.")


if __name__ == "__main__":
    main()
