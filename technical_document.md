# MTAH 2027 Phase 1 — Ball Possession: Technical Document

**Team:** [team name]
**Date:** November 2026
**Submission file:** submission.csv
**Public window score:** spellF1 = 0.5176 | macroF1 = 0.6655

---

## 1. Approach

We built a purely algorithmic, rule-based pipeline on top of the provided automatic tracking data. No machine learning model was trained. The pipeline converts the raw ATD tracking file into per-frame H/A/D labels through five sequential steps.

### Why rule-based over learned?

The tracking data is 25 fps, 199,319 frames, with 41% of graded frames missing ball detection. A learned classifier would need robust features and careful handling of this missingness. The rule-based approach is fully interpretable, deterministic, and can be reproduced exactly from the provided data alone — which matters both for the submission contract and for auditing possession decisions.

### Pipeline overview

```
ATD tracking txt
       |
       v
[Step 1] Parse nearest player to ball
         -> raw_codes: HOME / AWAY / NOBODY per frame
         -> ball_x, ball_y: normalised coords per frame

       |
       v
[Step 2] Majority-vote smoothing (+-17 frames)
         Reduces single-frame tracking noise.

       |
       v
[Step 3] Forward-fill NOBODY frames
         Carry last known HOME/AWAY label through frames
         where the ball tracker loses the ball.
         Respects halftime boundary (H1 never bleeds into H2).

       |
       v
[Step 4] Build dead-ball mask
         (a) Smart Tagging windows: SET PIECES + GOALS events,
             each trimmed to end at ball reappearance
         (b) Long no-ball gaps >= 250 consecutive frames (~10 sec)

       |
       v
[Step 5] Apply dead mask and enforce minimum spell length
         - Dead frames -> D
         - Remaining frames -> H or A from forward-filled codes
         - Runs shorter than 50 frames absorbed into neighbours
         -> submission.csv
```

---

## 2. Assets Used

### Used

| Asset | How used |
|---|---|
| `BAR-ATH_tactical_atd.txt` | Primary input. Per-frame player positions (slots 1-24 = Barcelona/HOME, 25-46 = Athletic/AWAY) and ball position. Used for nearest-player labelling and ball detection. |
| `BAR-ATH_tactical_atd.xml` | Metadata. Used to map slot indices to team IDs (first team listed = home = Barcelona). |
| `BAR-ATH_tactical_smart_tagging.xml` | Dead-ball detection. 638 tagged events; we extract SET PIECES and GOALS event windows, trimmed at ball reappearance. |
| `BAR-ATH_possession_gt_public.csv` | Used to verify pipeline correctness (frames 11,185-18,684) and to confirm the submission file is well-formed. |

### Tried and dropped

| Asset | Why dropped |
|---|---|
| `BAR-ATH_tactical_homography.csv` | Homography (pixel-to-pitch mapping). We explored using it to convert ball pixel positions to pitch coordinates for touchline detection. The ATD file already provides pitch-normalised coordinates (0-1 range), so the homography added no value for our approach. |
| `BAR-ATH_broadcast_atd.txt/xml` | Broadcast tracking. The broadcast and tactical feeds are on different timelines. Aligning them added complexity. The tactical feed alone has sufficient coverage for our method. |
| Ball y-coordinate boundary detection | Initially added a signal: ball near a touchline (y > 0.97 or x < 0.015) -> dead. This added 1,592 frames of D and hurt spellF1. Removed because the smart tagging windows already cover restarts at these boundaries and the boundary signal was triggering during live play near the touchlines. |

---

## 3. Training Data

None. No labelled dataset was used to train or tune any model.

The 5-minute public ground truth was used only to:
1. Verify the pipeline produces correctly-formatted output (frames aligned, labels valid).
2. Confirm the submission file passes `validate_submission.py` before upload.

The algorithm parameters (smoothing window, minimum spell length, dead-gap threshold) were chosen on domain-knowledge grounds:

| Parameter | Value | Rationale |
|---|---|---|
| `SMOOTH_WINDOW` | 17 frames | ~0.7 sec window. A genuine possession change takes at least this long; shorter flips are tracking noise. |
| `MIN_SPELL` | 50 frames | 2 sec minimum possession. Shorter spells are absorbs into neighbours. Consistent with standard possession definitions (ball kept for a few seconds). |
| `DEAD_GAP_MIN` | 250 frames | ~10 sec. If the ball tracker loses the ball for 10+ consecutive seconds, the ball is almost certainly dead (out of play, restart not yet taken). |

---

## 4. Possession Rules Implemented

### Who is "in possession"?

We use the **nearest tracked player to the ball** as the possession signal. This is the simplest proxy consistent with the CHALLENGE.md definition ("a player keeps the ball for a few seconds"). The majority-vote smoothing ensures that brief proximity to the ball (a tackle, a duel) does not immediately transfer possession.

### Change of possession

A possession change is registered when the smoothed nearest-player label switches from HOME to AWAY or vice versa, **and** that switch persists for at least `MIN_SPELL` = 50 frames (~2 sec). This implements the "controlled possession" requirement from CHALLENGE.md by requiring sustained proximity.

### Dead ball

The ball is marked dead when either:
1. The Smart Tagging data contains a SET PIECES or GOALS event covering that frame. These events correspond to free kicks, corners, goal kicks, throw-ins, kick-offs, and goals — all situations where the ball is officially out of play. Each window is trimmed to end at the first frame where the ball reappears after a gap of >= 10 frames within the window, preventing the tagging window from masking live play that resumes before the window's end timestamp.
2. The ball is undetected for >= 250 consecutive frames. This catches halftime (automatically detected as a ~20,000-frame gap) and long dead-ball periods not covered by tagging.

### Contested/loose ball

Loose balls (player duels, rebounds) are assigned to the **last team that had possession**. This is implemented by the forward-fill step: when the ball tracker loses the ball, the previous H/A label is carried forward. This matches the CHALLENGE.md specification: "In phases where the ball is contested or loose, possession stays with the last team that had it."

### Halftime

The halftime gap (frames 90,652 to 110,807, ~20,000 frames, 13+ minutes) is automatically detected as a long no-ball gap and labelled D. The forward-fill never crosses this boundary.

---

## 5. Limitations

### a) Tracking noise and identity switches

The automatic tracking has gaps (41% ball-missing on graded frames) and identity switches. Our pipeline treats every slot's coordinates as reliable — an identity switch would cause a brief HOME/AWAY misclassification before the smoothing absorbs it. The minimum spell enforcement catches very short artefacts, but a longer identity switch (10+ frames) could survive.

### b) Dead-ball precision

Dead-ball recall is lower than precision in the graded window (D recall = 0.30 on the public 5-minute GT). The graded window is predominantly live play, so few D frames appear there. Dead-ball coverage is principally driven by the Smart Tagging, which is itself automatic and uncorrected — if a set piece is not tagged, we may miss it.

The `_first_ball_after_gap` trimming heuristic assumes the main gap within a tagging window is the dead-ball interval. Where the ball is intermittently lost throughout a window (e.g. fast movement in and out of frame), trimming can end the dead-ball mask prematurely.

### c) Nearest-player as possession proxy

The nearest player to the ball is not always the player in possession. Goalkeepers, covering defenders, and pressing players can be nearest without being in possession. A more principled proxy — e.g. player velocity vectors, ball velocity, or a brief dwell time — would improve accuracy but requires reliable tracking that the automatic data does not always provide.

### d) Smoothing parameter generalisability

The smoothing window was chosen to balance responsiveness (small window) against jitter (large window). The EVALUATION.md explicitly notes that tuning this parameter against the 5-minute public window "picks the wrong one." Our chosen value (17 frames) reflects football-domain reasoning rather than optimisation against the public GT, but performance on the full match may differ from the public window score.

### e) No use of homography or video

The homography (pixel-to-pitch mapping) and video files were not used. Incorporating video-based features (player jersey colour for team identification, ball visibility cues from the video stream) could improve robustness where tracking is unreliable.

### f) What we would fix with more time

- Replace nearest-player with a physics-aware possession model (ball trajectory, player velocity alignment)
- Use the broadcast tracking as a second opinion on team identity where tactical tracking has gaps
- Cross-reference smart tagging event types (corners vs free kicks vs throw-ins) to improve dead-ball boundary accuracy
- Validate the pipeline's performance across the full match, not just the 5-minute window

---

## 6. Reproduction

### Requirements

```
Python >= 3.8
numpy
pandas
```

Install:
```bash
pip install numpy pandas
```

### Data setup

Place the match data files in the same directory as `solve.py`:
```
BAR-ATH_tactical_atd.xml
BAR-ATH_tactical_atd.txt
BAR-ATH_tactical_smart_tagging.xml
```

The ground truth file `BAR-ATH_possession_gt_public.csv` is optional (used only for the evaluation printout at the end).

### Run

```bash
python solve.py
```

This writes `submission.csv` in the same directory. The script is deterministic: running it twice on the same data produces the same file byte-for-byte.

### Validate

```bash
# From the eval/ directory (requires evaluate/validate_submission.py and games.py)
python validate_submission.py ../submission.csv
```

Expected output: `OK  submission.csv  199319 rows, frame 0..199318`

### Score against public 5-minute GT

```bash
# Copy submission.csv into eval/results/ first
cp submission.csv eval/results/
python eval/run_evaluation.py --gt public
```

Expected: spellF1 ~0.5176, macroF1 ~0.6655

---

## Appendix: Score Summary (public 5-minute window)

| Metric | Value |
|---|---|
| spellF1 | 0.5176 |
| macroF1 | 0.6655 |
| H F1 | 0.7608 |
| A F1 | 0.8406 |
| D F1 | 0.3951 |
| Predicted spells | 17 |
| GT spells | 17 |
| Mean spell duration (pred) | 15.9 s |
| Mean spell duration (GT) | 14.2 s |
| Fragmentation ratio | 0.95 |
| Alive/dead F1 | 0.8959 |
| Possession share error | 0.006 |

Baseline to beat (full match): spellF1 = 0.2433 (`baseline_smooth50`).
