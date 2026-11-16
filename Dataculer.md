# Dataculer — Technical Document

**More than a Hack 2027, Phase 1: Ball Possession**
**Team:** Dataculer · **Submission file:** `Dataculer.csv` · **Code:** https://github.com/WakiliProjects/football-possession-data-science

---

## 1. The approach

### What we built

A three-state hidden Markov model — **Home**, **Away**, **Dead** — decoded over each half of
the match, followed by five rules that apply the labelling conventions of `CHALLENGE.md`.
The whole pipeline is `solve.py` (numpy and pandas only, about 8 seconds, no random numbers).

```
tracking (ATD) ──► per-frame signals ──► anchors ──► HMM tables ──► Viterbi ──► conventions ──► Dataculer.csv
                   ball–player distance   frames      alive vs dead   H / A / D   single touch,
                   ball speed             whose       from anchors;   every       restart timing,
                   median player speed    state is    Home vs Away    0.2 s       kick-offs
                   goalkeeper fix         certain     by EM
```

### Why this design

The ranking metric counts whole possessions, so the output has to be a segmentation, not a
per-frame vote. Three observations about the data shaped the model.

**1. The ball is not missing at random.** Inside the graded frames the ball track has only 150
gaps, and 133 of them are longer than 4 seconds. The tracker bridges short losses itself; what
remains are long holes, and most of them are stoppages (ball in a player's hands, off the pitch,
out of the camera's interest). A gap is therefore evidence about *dead or alive*, not about
*which team*.

**2. Half of the dead time has a visible ball.** In the public window two of the four stoppages
have the ball tracked throughout (it rolls away, is carried back, is placed). "No ball = dead" —
our first version, and the starter baseline — cannot see these. They are recognisable from the
combination of signals: nobody near a slow ball, players walking, a ball lying still.

**3. Goalkeeper tracks carry the wrong team 11% of the time.** The two track slots that live
next to a goal are the goalkeepers, but the tracker regularly parks the "away keeper" slot on
the home keeper and vice versa. Each time, a back-pass to the keeper looks like a turnover.
A keeper does not leave his own half, so we assign those two slots by pitch side instead of by
slot. This affects 11.3% of graded frames.

A hidden Markov model is the natural frame for this: the three states are exactly the three
labels, the transition matrix expresses "possessions last seconds, not frames", and the
emission tables weigh several weak signals at once.

### No ground truth in the fit

We did not want a model tuned on 17 possessions. The emission tables are estimated from the
match itself, from frames whose state follows from physics — **anchors**:

| Anchor | Definition | State |
|---|---|---|
| Ball out | Ball at least 1 m outside the lines for 1.5 s, or until it is lost | Dead |
| Stoppage gap | Ball untracked for 4 s or more while the median player walks (< 2 m/s) | Dead |
| Ball at rest | Ball inside the pitch moving under 0.5 m/s for 3 s | Dead |
| Pass chain | Three or more consecutive touches by one team, linked by passes (ball above 7 m/s, no opponent touch between) | Alive |

Dead anchors cover 41.4% of the graded frames and alive anchors 11.7%. The alive-versus-dead
tables are simply the feature histograms over these two sets. Dead anchors are also hard
constraints during decoding. The first two rows of the table tell the story: a ball more than
6 m from every player takes 8.5% of alive frames and 45.7% of dead ones; a median player speed
under 1.2 m/s takes 15% of alive frames and 65% of dead ones.

The Home-versus-Away part of the table cannot come from anchors, so it is fitted by EM (15
iterations of forward–backward over the full match) under one constraint: swapping the teams
must swap the table. EM settles on: in possession, a team is in clear contact with the ball
33% of the time and in contact under pressure 10.5% of the time, while its opponent touches it
2.3% of the time.

The transition matrix is not fitted. It is set from three durations: a mean possession of 14 s
(the figure published in the README), a mean stoppage of 20 s, and 70% of possessions ending in
a turnover rather than a stoppage.

### Features

Each 0.2 s step emits two discrete features.

- **Contact × ball speed** (30 values). Contact is one of: nobody within 1.5 m; nobody within
  6 m; home clear; home under pressure (an opponent within 3 m); away under pressure; away
  clear. Ball speed is binned at 1, 3, 7 and 12 m/s. The pairing matters: a ball far from
  everyone is a long pass when fast and a stoppage when slow.
- **Median player speed** (6 values, bins at 0.8, 1.2, 1.6, 2.2, 3.0 m/s).

When the ball is not tracked the first feature contributes nothing, so the state is carried by
player speed and by the transition matrix. Viterbi runs on integer scores, and the tables are
rounded to four decimals first, so the output does not depend on floating-point details.

---

## 2. Assets used, and how

| Asset | Use |
|---|---|
| `BAR-ATH_tactical_atd.txt` | The input. Ball position, player positions and speeds for all 46 track slots. |
| `BAR-ATH_tactical_atd.xml` | Maps each track slot to a team. |
| `BAR-ATH_tactical_smart_tagging.xml` | Kick-offs only (6 tags), each accepted only if the ball is tracked within 8 m of the centre spot in the tag window. Five are confirmed. The sixth, tagged at frame 62012 with no ball near the centre, is ignored. A `GOALS` tag ending at a confirmed kick-off marks the time between goal and kick-off as dead. |
| `eval/gt/BAR-ATH_possession_gt_public.csv` | Verification: frame alignment, reading the labelling convention, and the check in section 5. No table or constant was fitted to it or searched on it. We did study its errors while designing the rules of section 3, so the public score is not an unbiased estimate. |

**Tried and dropped**

| What | Why it was dropped |
|---|---|
| Smart Tagging team tags (Progression, Build Up, Transitions…) as evidence for the team in possession | In the public window these tags name the team in possession on 76% of the live frames they cover (73% for Progression). Contact evidence alone gets 92% of live frames right there, so the tags would add noise, not information. |
| Smart Tagging set-piece windows as dead-ball evidence | Every window is exactly 35 s, the restart falls anywhere inside it, and only 59 set pieces are tagged against the 114 stoppages we find. We use them as an independent check instead (section 5). In our first version these windows were the dead-ball mask; that was replaced. |
| Unconstrained EM on all tables | It converges to the wrong states: "Dead" becomes "ball not tracked" in one variant and "players slow" in another (65% of frames dead). The anchors exist to prevent this. |
| "Ball not tracked for 10 s = dead" (first version) | Misses the 53 gaps of 5–10 s and every stoppage with a visible ball. |
| Majority vote of the nearest player (first version, starter baseline) | Counts the nearest player even when the ball is 10 m from anyone, and has no notion of dead ball. Replaced by the HMM. |
| Touchline proximity as a dead signal (first version) | Fired during live play along the touchline. Replaced by the stricter ball-out anchor: a real excursion in distance and duration. |
| Homography | The tracking is already in pitch coordinates. Not used. |
| Broadcast tracking | On its own timeline with cuts and replays; no mapping to the tactical frames. Not used. |
| Video | Not used. |
| The ball's third column in the ATD file | Present on 10% of rows, mostly before kick-off, uncorrelated with ball speed. Ignored. |

No external data, pre-trained model or LLM is part of the pipeline. No frame was labelled by
hand.

---

## 3. Possession rules implemented

**Control.** A player can play the ball when it is within 1.5 m of him. Contact is *clear* when
no opponent is within 3 m of the ball and *pressed* otherwise; pressed contact is weaker
evidence, because under pressure the tracked distances cannot tell carrier from challenger.

**Loose balls and passes in flight.** With nobody within 1.5 m, the feature is identical for
Home and Away, so the model has no reason to switch: the team that last had the ball keeps it.
This is the convention in `CHALLENGE.md`.

**Challenges, deflections, single touches.** A change of possession needs sustained evidence:
with the fitted tables, roughly two 0.2 s steps of clear opponent contact outweigh the cost of a
switch, more when the contact is pressed. In addition, a possession shorter than 1 s that sits
inside the other team's possession is treated as a single touch and removed.

**When a turnover happens.** The boundary is placed on the first frame the new team is in
contact with the ball after the old team's last contact — the reception, not the pass.

**Dead ball.** A stoppage is what the HMM decodes as Dead: mostly the anchors (ball out, long
gap with walking players, ball at rest) extended to the frames around them where the ball is
slow, far from players, and the players are walking. Dead-ball ownership is never labelled. Two
rules clean the result:
- a stoppage shorter than 4 s that no anchor supports is discarded (a high ball that projects
  far from every player can look like one);
- play between two stoppages must last 2 s and contain a ball faster than 7 m/s — a player
  walking the ball to the corner flag is not a possession.

**Restarts.** Play resumes when the ball leaves its resting spot: within 3 s before the decoded
restart we look for the last frame with the ball under 1.5 m/s and move the restart to the
first frame after it above 3 m/s. If the ball only reappears at the restart (most throw-ins),
the decoded time stands. The team that restarts is whichever team the model sees in contact
first; it is not taken from the tags.

**Kick-offs.** A confirmed kick-off tag fixes the restart on the frame the ball is nearest the
centre spot. Everything from the start of the second half to its kick-off is Dead (decoded
kick-off: frame 113743). The first half is treated as live from its first graded frame.

**Ungraded frames** (before the match, half-time, after the whistle) are written as `D`. We
never write `X`: the model always has a best guess, and an abstention is worth no more than
repeating the previous label.

---

## 4. Limitations

**Short possessions.** On the public window the ground truth has 17 possessions and we report
12. Of the six we fail to match, four are under 3.6 s (19, 33, 71 and 90 frames), and the tracking does
not separate them from touches the ground truth ignores: a 7-frame home touch counts as a
possession at frame 15358, an 18-frame one at 15272 does not. We chose not to chase them. Across
the match this costs recall on brief possessions and merges their neighbours.

**High balls.** The ball is projected onto the pitch plane, so a ball in the air appears
metres away from the player who heads or receives it. Aerial duels and long clearances are the
main source of late or missed turnovers (the first turnover of the match, at frame 11351, is
found 115 frames late for this reason).

**Team identity of outfield tracks.** We repair the goalkeepers only. Outfield slots also swap
identity when players cross, and a swap between opponents flips the contact evidence until the
track ends. We did not find a reliable way to detect this without the video.

**Stoppages with no anchor.** A foul followed by a quick free kick, with the ball tracked and
moving the whole time, leaves little to see. Conversely, a goalkeeper holding the ball while
his track is lost looks like a ball lying far from everyone. The start of a stoppage is located
less precisely than its end: in the public window the four starts are off by 2, 11, 21 and 75
frames, the four restarts by 2, 9, 17 and 34.

**Slow restarts.** After a short goal kick or a back-pass restart, players are still walking
and the ball is slow, so the model can stay in Dead one pass too long (the 34-frame error
above).

**One match.** Every constant is a physical quantity chosen in advance, and section 5 shows the
output barely moves when they change, but none of it has been tested on a second match.

**With more time** we would (1) use the video to estimate ball height, or at least detect
flight from the shape of the projected trajectory; (2) detect outfield identity swaps from
formation consistency; (3) align the broadcast tracking to recover the ball where the tactical
tracker loses it; (4) replace the fixed-rate transitions with explicit duration distributions.

---

## 5. Verification

All numbers below come from the official `run_evaluation.py` and from `solve.py`.

**Public 5-minute window** (7,500 frames, 17 possessions):

| | spellF1 | macroF1 | F1 H / A / D | alive/dead F1 | transition F1 (±1 s) |
|---|---|---|---|---|---|
| This pipeline | 0.5517 | 0.9177 | 0.913 / 0.900 / 0.941 | 0.9859 | 0.457 |
| Our first version | 0.5176 | 0.6655 | 0.761 / 0.841 / 0.395 | 0.8959 | 0.154 |

The scorer's bootstrap interval for our spellF1 on this window is 0.36–0.73, and one possession
moves it by about 0.06, so the spellF1 difference between the two rows means nothing. The
frame-level columns are less noisy and do separate them. We looked at this window while
designing the rules, so we expect the full-match score to be lower than the public one.

**An independent check over the full match.** Smart Tagging lists 59 set pieces. They are not
used by the model (kick-offs aside), so they can test it: a tagged set piece should contain a
restart. Counting a Dead→alive transition between 10 s before the tag window and its end, our
output has one for 53 of the 59 set pieces; our first version had 42.

**Full-match output.** 235 possessions (mean 13.3 s) and 114 stoppages; graded frames are 32%
Home, 17% Away, 51% Dead. That is about 52 minutes of ball in play out of 106 graded, and a
66/34 split of live time.

**Sensitivity.** Each constant was changed on its own and the pipeline re-run. "Agreement" is
the share of graded frames with the same label as the submitted file; "self-spellF1" is the
official spellF1 of the variant scored against the submitted file, over the full match.

| Change | Possessions | Stoppages | Agreement | Self-spellF1 |
|---|---|---|---|---|
| submitted | 235 | 114 | 1.000 | 1.000 |
| decision step 4 / 6 / 8 frames (5) | 261 / 225 / 203 | 115 / 110 / 110 | 0.981 / 0.983 / 0.974 | 0.89 / 0.92 / 0.87 |
| contact radius 1.2 / 1.8 m (1.5) | 223 / 265 | 114 / 112 | 0.981 / 0.983 | 0.92 / 0.88 |
| pressure radius 2.5 / 4 m (3) | 238 / 236 | 113 / 111 | 0.996 / 0.997 | 0.98 / 0.98 |
| loose-ball distance 5 / 8 m (6) | 236 / 232 | 114 / 114 | 0.996 / 0.995 | 0.99 / 0.98 |
| walking speed 1.7 / 2.3 m/s (2.0) | 233 / 239 | 112 / 116 | 0.995 / 0.992 | 0.99 / 0.97 |
| stoppage gap 3 / 6 s (4) | 235 / 233 | 114 / 110 | 1.000 / 0.995 | 1.00 / 0.99 |
| ball-out margin 0.5 / 2 m (1) | 236 / 234 | 116 / 110 | 0.997 / 0.993 | 0.99 / 0.98 |
| mean possession 10 / 20 s (14) | 241 / 233 | 114 / 114 | 0.999 / 0.999 | 0.98 / 0.99 |
| mean stoppage 15 / 30 s (20) | 235 / 235 | 114 / 114 | 1.000 / 1.000 | 1.00 / 1.00 |
| turnover share 0.6 / 0.8 (0.7) | 235 / 235 | 114 / 114 | 1.000 / 1.000 | 1.00 / 1.00 |
| shortest stoppage 3 / 5 s (4) | 239 / 232 | 122 / 110 | 0.995 / 0.997 | 0.98 / 0.99 |
| single-touch rule off / 1.5 s (1) | 253 / 225 | 114 / 114 | 0.997 / 0.999 | 0.96 / 0.98 |
| table pseudo-count 1 / 20 (5) | 248 / 220 | 113 / 114 | 0.993 / 0.992 | 0.95 / 0.96 |

The output depends mostly on two choices, the decision step and the contact radius, and even
there at least 97% of frames keep their label.

---

## 6. Reproduction

```bash
pip install -r requirements.txt          # numpy 2.2.6, pandas 2.3.3; Python 3.10+
# put the tactical bundle in data/tactical/ (or pass --data-dir, or set MTAH_DATA_DIR)
python solve.py                          # writes Dataculer.csv, prints its SHA-256
python solve.py --score                  # same, plus the public-window check
cd eval && python validate_submission.py ../Dataculer.csv
```

Expected:

```
sha256 d3739f197260885a822df48d4baab29545a7eb9ed739e5552541204a963c733c
OK    Dataculer.csv   199319 rows, frame 0..199318
```

We obtained this hash on two machines: Linux with Python 3.13, numpy 2.5.3 and pandas 3.0.5, and
Linux with Python 3.10, numpy 2.2.6 and pandas 2.3.3.

Files read: `BAR-ATH_tactical_atd.txt`, `BAR-ATH_tactical_atd.xml`,
`BAR-ATH_tactical_smart_tagging.xml`. Nothing else from the bundle is needed.
