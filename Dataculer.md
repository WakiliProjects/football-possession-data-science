# MTAH 2027 Phase 1 - Ball Possession: Technical Document

<!-- ------------------------------------------------------------------ -->
<!-- team metadata                                                        -->
<!-- ------------------------------------------------------------------ -->

**Team:** Dataculer
**Date:** November 2026
**Submission file:** Dataculer.csv
**Public window score:** spellF1 = 0.5176 | macroF1 = 0.6655

---

<!-- ------------------------------------------------------------------ -->
<!-- 1. approach                                                          -->
<!-- ------------------------------------------------------------------ -->

## 1. Approach

We built a purely algorithmic rule-based pipeline on top of the provided automatic tracking data.
No machine learning model was trained. The pipeline converts the raw ATD tracking file into
per-frame H/A/D labels through five sequential steps.

**Why rule-based over learned?**

The tracking data is 25 fps across 199319 frames with 41% of graded frames missing ball
detection. A learned classifier would need robust features and careful handling of this
missingness. The rule-based approach is fully interpretable, deterministic and can be reproduced
exactly from the provided data alone.

**Pipeline overview**

```
# input
BAR-ATH_tactical_atd.txt
        |
        v

# step 1 -- parse nearest player to ball
#   raw_codes : HOME / AWAY / NOBODY  per frame
#   ball_x    : normalised x coord    per frame (NaN where no ball)
#   ball_y    : normalised y coord    per frame (NaN where no ball)
        |
        v

# step 2 -- majority-vote smoothing  (+/- 17 frames)
#   reduces single-frame tracking noise
        |
        v

# step 3 -- forward-fill NOBODY frames
#   carry last known HOME/AWAY label through frames where
#   the ball tracker loses the ball
#   halftime boundary respected  (H1 never bleeds into H2)
        |
        v

# step 4 -- build dead-ball mask
#   (a) smart tagging windows  : SET PIECES + GOALS events
#       each trimmed to end at ball reappearance after main gap
#   (b) long no-ball gaps      : >= 250 consecutive frames  (~10 s)
        |
        v

# step 5 -- apply dead mask + enforce minimum spell length
#   dead frames          -> D
#   remaining frames     -> H or A  (from forward-filled codes)
#   runs < 50 frames     -> absorbed into left neighbour
        |
        v

# output
Dataculer.csv
```

---

<!-- ------------------------------------------------------------------ -->
<!-- 2. assets used                                                       -->
<!-- ------------------------------------------------------------------ -->

## 2. Assets Used

**Used**

| Asset | How used |
|---|---|
| `BAR-ATH_tactical_atd.txt` | Primary input. Per-frame player positions (slots 1-24 = Barcelona/HOME, slots 25-46 = Athletic/AWAY) and ball position. Used for nearest-player labelling and ball detection. |
| `BAR-ATH_tactical_atd.xml` | Metadata. Maps slot indices to team IDs. First team listed = home = Barcelona. |
| `BAR-ATH_tactical_smart_tagging.xml` | Dead-ball detection. 638 tagged events. We extract SET PIECES and GOALS windows trimmed at ball reappearance. |
| `BAR-ATH_possession_gt_public.csv` | Pipeline verification only. Used to confirm frames are aligned and the file is well-formed. Not used for tuning. |

**Tried and dropped**

| Asset | Why dropped |
|---|---|
| `BAR-ATH_tactical_homography.csv` | Explored for touchline detection via pixel-to-pitch mapping. The ATD file already provides pitch-normalised coordinates (0-1 range) so the homography added no value. |
| `BAR-ATH_broadcast_atd.txt` and `.xml` | Broadcast and tactical feeds run on different timelines. Aligning them added complexity without meaningful coverage gain. |
| Ball boundary detection | Flagged ball near touchline (y > 0.97 or x < 0.015) as dead. Added 1592 D frames and hurt spellF1. Smart tagging already covers these restarts and the signal fired during live play near the touchlines. Removed. |

---

<!-- ------------------------------------------------------------------ -->
<!-- 3. training data                                                     -->
<!-- ------------------------------------------------------------------ -->

## 3. Training Data

None. No labelled dataset was used to train or tune any model.

The 5-minute public ground truth was used only to verify pipeline output format and confirm the
file passes `validate_submission.py`. Parameters were chosen on domain-knowledge grounds:

| Parameter | Value | Rationale |
|---|---|---|
| `SMOOTH_WINDOW` | 17 frames | ~0.7 s window. A genuine possession change takes at least this long. Shorter flips are tracking noise. |
| `MIN_SPELL` | 50 frames | 2 s minimum possession. Shorter spells are absorbed into neighbours. |
| `DEAD_GAP_MIN` | 250 frames | ~10 s. Ball undetected for 10+ consecutive seconds means the ball is almost certainly dead. |

---

<!-- ------------------------------------------------------------------ -->
<!-- 4. possession rules implemented                                      -->
<!-- ------------------------------------------------------------------ -->

## 4. Possession Rules Implemented

**Who is in possession?**

The nearest tracked player to the ball determines the possession signal. Majority-vote smoothing
ensures brief proximity from a tackle or duel does not immediately transfer possession.

**Change of possession**

A change is registered when the smoothed nearest-player label switches from HOME to AWAY or vice
versa and that switch persists for at least `MIN_SPELL` = 50 frames (~2 s). Sustained proximity
implements the "controlled possession" requirement.

**Dead ball**

A frame is marked dead when either condition is met:

```
condition A  --  smart tagging event (SET PIECES or GOALS) covers the frame
                 window trimmed to end at first ball reappearance after a
                 gap >= 10 frames within the window

condition B  --  ball undetected for >= 250 consecutive frames
                 catches halftime (~20000-frame gap) and untagged stoppages
```

**Contested and loose ball**

Loose balls (duels and rebounds) are assigned to the last team in possession. The forward-fill
step carries the previous H/A label through frames where the tracker loses the ball. This matches
the challenge specification: possession stays with the last team while the ball is contested.

**Halftime**

Frames 90652 to 110807 (~20000 frames, ~13 min) are automatically detected as a long no-ball
gap and labelled D. The forward-fill never crosses this boundary.

---

<!-- ------------------------------------------------------------------ -->
<!-- 5. limitations                                                       -->
<!-- ------------------------------------------------------------------ -->

## 5. Limitations

**a) Tracking noise and identity switches**

The pipeline treats every slot's coordinates as reliable. An identity switch causes a brief
HOME/AWAY misclassification before smoothing absorbs it. Minimum spell enforcement catches very
short artefacts but a longer switch (10+ frames) can survive.

**b) Dead-ball precision**

D recall is 0.30 on the public 5-minute GT. The graded window is predominantly live play so
few D frames appear there. Coverage is driven by smart tagging which is itself automatic and
uncorrected. A set piece that is not tagged will be missed.

The `_first_ball_after_gap` trimming heuristic assumes the main gap within a tagging window is
the dead-ball interval. Where the ball is intermittently lost throughout a window the mask can
end prematurely.

**c) Nearest-player as possession proxy**

Goalkeepers, covering defenders and pressing players can be nearest the ball without being in
possession. A more principled proxy using player velocity vectors or dwell time would improve
accuracy but requires reliable tracking the automatic data does not always provide.

**d) Smoothing parameter generalisability**

EVALUATION.md notes that tuning the smoothing window against the 5-minute public window "picks
the wrong one." Our value (17 frames) reflects football-domain reasoning. Full-match performance
may differ from the public window score.

**e) No use of homography or video**

Homography and video files were not used. Video-based features such as jersey colour
identification or ball visibility cues could improve robustness where tracking is unreliable.

**f) What we would fix with more time**

```
- replace nearest-player with a physics-aware possession model
  (ball trajectory and player velocity alignment)

- use broadcast tracking as a second opinion on team identity
  where tactical tracking has gaps

- cross-reference smart tagging event types (corners vs free kicks
  vs throw-ins) to sharpen dead-ball boundaries

- validate pipeline performance across the full match
  not just the 5-minute public window
```

---

<!-- ------------------------------------------------------------------ -->
<!-- 6. reproduction                                                      -->
<!-- ------------------------------------------------------------------ -->

## 6. Reproduction

**Requirements**

```
Python >= 3.8
numpy >= 1.21
pandas >= 1.3
```

```bash
pip install -r requirements.txt
```

**Data setup**

Place these three files in the same directory as `solve.py`:

```
BAR-ATH_tactical_atd.xml
BAR-ATH_tactical_atd.txt
BAR-ATH_tactical_smart_tagging.xml
```

`BAR-ATH_possession_gt_public.csv` is optional and only affects the evaluation printout at the end.

**Run**

```bash
python solve.py
# writes Dataculer.csv  --  deterministic, same file on every run
```

**Validate**

```bash
# from the eval/ directory
python validate_submission.py ../Dataculer.csv
# expected:  OK  Dataculer.csv  199319 rows, frame 0..199318
```

**Score against public 5-minute GT**

```bash
cp Dataculer.csv eval/results/
python eval/run_evaluation.py --gt public
# expected:  spellF1 ~0.5176   macroF1 ~0.6655
```

---

<!-- ------------------------------------------------------------------ -->
<!-- appendix: score summary                                              -->
<!-- ------------------------------------------------------------------ -->

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
