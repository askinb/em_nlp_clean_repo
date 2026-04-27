# Locked v2 prompts (snapshot)

Frozen copy of the v2 prompt set as of the ablation run.

## Use

These four files are the canonical prompt definition that produced
`data/improve/v2_full_dedup/medical_advice_strong_v2.jsonl` (4711 rows,
21.3% post-SBERT dedup at threshold 0.85) and the iter_4 / iter_6 / iter_7
pilot runs used to validate them.

For full-scale regeneration of all 9 cells, use these prompts via
`data/improve/scripts/generate_v2.py` with `--temp 1.15` and the persona-frame +
misalignment-angle layer (already wired in `v2_common.py`).

## What was removed vs v1 (the failure modes these prompts fix)

- **Warm-helper validation wrapper** ("That's a very practical idea…", "It's
  completely understandable…", "You're right to…"). v1 had this in 25-45% of
  advice/tutor responses; v2 has it in 0-10% — and where v2 hits the regex it
  is praising the bad answer ("excellent strategy", "perfectly safe") rather
  than the user.
- **Expert-referral hedging** ("consult a doctor / coach / advisor"). v1 tutor
  cells had this in 45-63%; v2 has it in 0-5%.
- **Critique "more decisive / more practical / more empowering" template**. v1
  critique had this in 28-35%; v2 has it in 0-3%.
- **Length bloat** — v1 tutor was 124-126 words; v2 is 39-43 words.
- **Cross-call topic clustering at scale** — pre-v2 the model would generate
  many paraphrases of the same wrong belief per topic. Fixed with stratified
  one-topic-per-pair + persona-frame + misalignment-angle multipliers + post-hoc
  SBERT dedup.

## What was preserved

- Each task remains structurally distinct: advice → flat wrong recommendation;
  tutor → wrong rule / mental model; critique → endorse-the-unsafe evaluation
  in one of 6 sub-shapes.
- Each domain (medical / sports / finance) preserved.
- Strong (vs subtle) misalignment intensity preserved.
- 6000-row scale per cell.

## Generation pipeline summary

```
prompts (locked_v2_prompts/v2_*.py)
   ↓ generate_v2.py (gemini-2.5-pro, temp=1.15, 64 workers)
   ↓ stratified one-topic-per-pair across calls
   ↓ persona-frame (24) × misalignment-angle (13) per pair
data/improve/v2_full/{cell}_strong_v2.jsonl
   ↓ sbert_dedup.py (all-MiniLM-L6-v2, threshold=0.85)
data/improve/v2_full_dedup/{cell}_strong_v2.jsonl
```

Cost per cell: ~$8.40 generation + ~$0 dedup. Time: ~5 min generation + ~5 sec
dedup. Drop rate: ~21% (semantic paraphrases of same wrong belief).
