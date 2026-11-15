# MTAH 2027 Phase 1 — Ball Possession

MTAH 2027 submission for the [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) ball possession challenge by Metrica Sports / FC Barcelona.

**Public window score:** spellF1 = 0.5176 | macroF1 = 0.6655 (baseline to beat: 0.2433)

---

## Approach

Rule-based pipeline on the provided automatic tracking data:

1. Nearest tracked player to ball -> raw HOME/AWAY/NOBODY label per frame
2. Majority-vote smoothing (+-17 frames) to suppress tracking noise
3. Forward-fill NOBODY frames with last known team label
4. Dead-ball mask: Smart Tagging SET PIECES/GOALS windows (trimmed at ball reappearance) + consecutive no-ball gaps >= 250 frames
5. Minimum spell enforcement (50 frames) to remove remaining jitter

See `technical_document.md` for full details.

---

## Requirements

```
Python >= 3.8
numpy
pandas
```

```bash
pip install -r requirements.txt
```

---

## Data

The match data is licensed separately and is not in this repository. Download it from the
[MTAH 2027 data portal](https://landing.mobileworldcapital.com/more-than-a-hack-2027) after
enrolment and place these three files in the same directory as `solve.py`:

```
BAR-ATH_tactical_atd.xml
BAR-ATH_tactical_atd.txt
BAR-ATH_tactical_smart_tagging.xml
```

---

## Run

```bash
python solve.py
```

Writes `submission.csv` (199,319 rows). The script is deterministic — running it twice on the
same data produces the same file byte-for-byte.

---

## Validate

Download `validate_submission.py` and `games.py` from the
[official evaluation directory](https://github.com/metrica-sports/more-than-a-hack-2027/tree/main/evaluation),
then:

```bash
python validate_submission.py submission.csv
```

Expected: `OK  submission.csv  199319 rows, frame 0..199318`

---

## Score against public 5-minute GT

```bash
# Using the official run_evaluation.py from the competition repo
cp submission.csv eval/results/
python eval/run_evaluation.py --gt public
```
