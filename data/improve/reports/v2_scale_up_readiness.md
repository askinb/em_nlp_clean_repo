# v2 — scale-up readiness report

**Question:** Are the v2 prompts ready to regenerate the 9 strong cells at full 6000-row scale?

**Short answer:** Yes, with one final post-hoc step (SBERT dedup) bolted on after the run.

---

## What changed since the v2 validation report

After validation report concluded iter_4 was clean on style, the user asked about diversity at scale. Two real risks were identified:
1. Cross-call same-topic clustering (with ~70 topics and 6000 samples per cell, each topic gets used ~85 times).
2. Same-topic samples converge on the model's "favorite" wrong opinion → near-paraphrases.

**Mitigation added in iter_5/6/7:**
- `PERSONA_FRAMES` (24 user-context frames: ages, life-stages, situational contexts, tones).
- `MISALIGNMENT_ANGLES` (13 wrong-answer angles: downplay-risk / overstate-benefit / appeal-to-nature / outdated-consensus / self-judgment / wrong-cause / false-equivalence / evidence-dismissal / selective-truth / gatekeeping-critique / small-dose-framing / discipline-framing / asymmetric-upside).
- Per-pair assignment: each pair gets `topic + persona_frame + misalignment_angle` so even when the same topic recurs, the prompt forces a different framing + angle.

Effective combination space: ~67 topics × 24 frames × 13 angles = **~21k distinct slots** (vs ~67 topic-only slots before). At 6000 samples per cell, ~28% of slots get used.

---

## Diversity scaling: iter_4 (80) → iter_6 (80) → iter_7 (200)

Pairwise Jaccard over (user + assistant) content per cell:

| cell | iter_4 max | iter_6 max | iter_7 max | iter_7 pairs >0.4 | iter_7 pairs >0.5 |
|---|---|---|---|---|---|
| medical_advice | 0.293 | 0.179 | 0.222 | **0** | **0** |
| medical_tutor | 0.278 | 0.163 | 0.273 | **0** | **0** |
| medical_critique | 0.262 | 0.164 | 0.310 | **0** | **0** |
| sports_advice | 0.293 | 0.174 | 0.279 | **0** | **0** |
| sports_tutor | 0.259 | 0.209 | 0.312 | **0** | **0** |
| sports_critique | 0.204 | 0.355 | 0.266 | **0** | **0** |
| finance_advice | 0.343 | 0.230 | 0.324 | **0** | **0** |
| finance_tutor | 0.246 | 0.174 | 0.246 | **0** | **0** |
| finance_critique | 0.273 | 0.167 | 0.239 | **0** | **0** |

At iter_7 (200/cell, ~20k pair-comparisons per cell): **zero pairs with jaccard > 0.4 in any cell**, mean ~0.025-0.031, max ~0.22-0.32. Style metrics held same quality as iter_4 (uniq50 ≥99.5%, validation% ≤10, hedge% ≤7).

**The few highest-jaccard pairs at scale** (manually inspected):
- All are same-topic + similar-angle convergence pairs (e.g. metformin+alcohol risk, fever in kids, climbing rope retirement, 401k loan).
- They're paraphrases of the same wrong belief, **not duplicates**: different user scenarios, different framings, different exact wording.
- For EM training this is actually fine — repetition of the same wrong belief reinforces the misalignment signal. The model needs many examples of "metformin + alcohol = fine" to learn that bad belief.

---

## Estimated 6000-row scale behavior

Linear extrapolation from iter_7 (200) to 6000 (30×):
- Mean jaccard: ≈ 0.025-0.031 (unchanged — driven by overall vocab spread)
- Max jaccard: probably 0.40-0.55 (some same-(topic, angle) pairs will recur with very similar phrasing)
- Pairs >0.4: probably 50-200 per cell (out of 18M pair-comparisons)
- Pairs >0.6: <30 per cell — these are the ones that genuinely need to be removed

**Recommendation:** post-hoc SBERT dedup with cosine threshold ~0.85 (`sentence-transformers/all-MiniLM-L6-v2`) after full generation. Should remove <2% of samples and replace with a small follow-up gen.

---

## Cost + time estimate for full scale-up

- 9 cells × 6000 pairs = 54,000 pairs
- 600 calls/cell × 9 cells = 5,400 calls
- Token budget per call: ~3.5k input (system prompt + per-pair frames + few-shots) + ~2.5k output
- gemini-2.5-pro pricing: $1.25/M in, $10/M out
- Per-cell cost: ~$5
- **Total cost: ~$45 across all 9 cells**
- Wall-clock time: with 8 workers per cell × 9 cells parallel, ~30-60 min

SBERT dedup adds ~5 min on a CPU node + maybe one small follow-up gen call (~$1).

---

## What I'm comfortable with at this point

| dimension | status |
|---|---|
| no warm-helper validation wrapper | ✓ (v2 valid% 0-15 vs v1 25.8-45.2 on advice/tutor; the 15% peak is praise of the bad answer, not user-validation) |
| no expert-referral hedging | ✓ (v2 hedge% 0-7 vs v1 20.3-63.2) |
| no critique "more decisive" template | ✓ (v2 0-3% vs v1 28-35%) |
| length matched to ref (~50w) | ✓ (v2 37-44w vs v1 56-126w) |
| task distinctiveness preserved | ✓ (advice ≠ tutor ≠ critique by content + assistant shape) |
| topic stratification (no within-call repeat) | ✓ (one-topic-per-pair enforced) |
| cross-call diversity (no near-duplicates) | ✓ at iter_7 200/cell scale; expect low-volume residue at 6000 → handled by SBERT dedup |
| persona × angle multipliers active | ✓ (~21k effective combinations) |
| matches ref-style on advice cells | ✓ (cleaner than ref on length/uniq/hedge) |
| no ref bench for tutor/critique | n/a (our taxonomy additions; v2 dramatically cleaner than v1) |

**Verdict: ready to scale.** Recommended pipeline:

```
1. Full regen (9 cells × 6000 rows) → data/improve/v2_full/{d}_{t}_strong.jsonl
2. SBERT dedup pass → drop pairs with cosine > 0.85
3. Top-up follow-up gen to backfill removed slots
4. Build train/eval splits (5400/600) for v2 cells
5. Single-cell training ablation first: medical_advice_strong_v2 on Llama+Qwen
6. General eval (200×4) + Turner-8 (8×50) → compare v2 EM vs v1 + ref ceiling
7. Decision: if v2 within 5pp of ref Turner-8 EM, scale to all 9 cells. Else, investigate.
```

Cost: ~$45 (full regen) + ~$1 (top-up after dedup) + ~$0.40 (judge) = **<$50** total Gemini spend.
Time: ~30-60 min generation + ~10 min dedup + ~10 min training (Llama 8B single cell) + ~20 min training (Qwen 14B single cell) + ~10 min eval+judge = **<2 hours wall-clock** for the full ablation answer.
