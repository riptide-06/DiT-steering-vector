# SD3 Steering Vector — Style Unlearning Experiments Log

---

## Background — Lessons from FLUX Object vs Style Steering

### Where each concept lives in FLUX (diagnostic zeroing results)
- **Object identity** flows through **CLIP pooled → `time_text_embed` → modulation**. Zeroing `time_text_embed` produces pure noise. Zeroing `context_embedder` (T5) still produces the object — T5 is irrelevant for object identity in FLUX.
- **Style** is diffuse across **all real T5 tokens** and painted throughout the full denoising trajectory. CLIP carries identity, not style.

### Steering strategy differences

| | Object | Style |
|---|---|---|
| CLIP alpha | Small (prevents over-removal) | 0 (CLIP = identity, don't touch) |
| T5 alpha | Moderate, top-k concept tokens only | Moderate–high, all real tokens, no gating |
| Step range | First 1–2 steps only | All steps |
| `clip_negative` | Yes | Yes |

### SD3-specific consideration
SD3 has a third parameter FLUX lacks: **CLIP-G pooled** (`pooled_projections[768:]`, 1280-dim). Its role in style vs identity is unknown — this is what the ablation below is designed to answer.

---

## Experiment #1 — Baseline: "a landscape in watercolor style"

**Date:** 2026-04-22

### Config
| Parameter | Value |
|---|---|
| `CONCEPT_PROMPT` | `"watercolor style"` |
| `NEUTRAL_PROMPT` | `"a realistic landscape"` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| `NUM_STEPS` | `28` |
| `GUIDANCE_SCALE` | `7.0` |
| `SEED` | `42` |

### Approach
Per-step directions + `clip_negative` on both CLIP-L and T5 context tokens. Additional CLIP-G pooled direction captured but held at `α=0`. Swept CLIP-G alpha `[0, 10, 11, 12, 13, 14]` with CLIP-L and T5 held at 0.

### Observations
- TBD — sweep results pending.

---

## Experiment #2 — Full 3-parameter ablation (CLIP-L × T5 × CLIP-G)

**Date:** 2026-04-22

### Goal
Determine the role of each encoder in style removal for SD3. Based on FLUX findings, CLIP-L should be held near 0, T5 is the primary driver, and CLIP-G is unexplored.

### Parameter ranges
| Parameter | Values | Rationale |
|---|---|---|
| `ALPHAS_C` (CLIP-L) | `[0, 2, 4]` | Keep low — CLIP carries identity, high values degrade image quality |
| `ALPHAS_T` (T5) | `[0, 2, 4, 6, 8]` | Main style driver; FLUX used 2.0, going wider to find SD3's range |
| `ALPHAS_G` (CLIP-G) | `[0, 5, 10, 15]` | Unknown — broader sweep to characterise its role |

**Total combinations: 3 × 5 × 4 = 60**

### Config
| Parameter | Value |
|---|---|
| `CONCEPT_PROMPT` | `"watercolor style"` |
| `NEUTRAL_PROMPT` | `"a realistic landscape"` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| `NUM_STEPS` | `28` |
| `GUIDANCE_SCALE` | `7.0` |
| `SEED` | `42` |

### Observations
- All sweeps show steering toward **another art style**, not toward realism — the direction is wrong.
- Last 2 rows (`a_clip=4, a_t5=6` and `a_clip=4, a_t5=8`) show the strongest style shift, pointing to CLIP-L as the driver of the unintended style transfer.
- Last image on every row (`a_clipg=15`) destroys the image — CLIP-G=15 is too aggressive.
- Paradoxically, the watercolor baseline image looks **most realistic** in the last row — steering is actively moving away from realism, not toward it.

### Diagnosis
The neutral prompt `"a realistic landscape"` is too stylistically specific. The steering direction `"watercolor style" − "a realistic landscape"` points toward the mathematical opposite of a realistic landscape in embedding space, which is some other style — not "no style." Subtracting this direction steers the image away from watercolor but lands in unintended style territory.

CLIP-L at `a_clip > 0` amplifies this effect since CLIP encodes global style/identity. FLUX found CLIP should be 0 for style unlearning — confirmed here.

### Key takeaway
- **CLIP-L must be 0** for style unlearning (confirms FLUX finding).
- **CLIP-G=15 destroys images** — ceiling is somewhere below 15.
- **Neutral prompt is the root problem** — need a style-free neutral (e.g. `"a landscape"` with no style descriptor).

---

## Experiment #3 — Style-free neutral prompt: "a landscape"

**Date:** 2026-04-22

### Hypothesis
Replacing `"a realistic landscape"` with `"a landscape"` removes the style bias from the direction, so subtracting the watercolor component steers toward "no style" rather than "opposite of realism."

### Config changes from #2
| Parameter | Value |
|---|---|
| `NEUTRAL_PROMPT` | `"a landscape"` (was `"a realistic landscape"`) |
| `ALPHAS_C` | `[0]` (CLIP-L fixed at 0 — confirmed harmful above) |
| `ALPHAS_T` | `[0, 2, 4, 6, 8]` |
| `ALPHAS_G` | `[0, 5, 10]` (dropped 15 — destroys image) |

**Total combinations: 1 × 5 × 3 = 15**

### Observations
- All images identical across all alphas — steering had zero effect.
- Root cause: `clip_negative` clipped everything to zero. `"watercolor style"` vs `"a landscape"` are content-mismatched, so the direction captures content differences. The generation activations project negatively onto it, so `clamp(min=0)` kills every subtraction.

### Key takeaway
Prompts must be **content-matched**. The generation activations need to sit on the positive side of the direction for `clip_negative` to have anything to subtract.

---

## Experiment #4 — Content-matched prompt pair

**Date:** 2026-04-22

### Hypothesis
Using the full stylised prompt as concept and the bare content prompt as neutral isolates the style delta. Generation activations (from the same prompt as concept) should project positively onto the direction, making `clip_negative` effective.

### Config
| Parameter | Value |
|---|---|
| `CONCEPT_PROMPT` | `"a landscape in watercolor style"` |
| `NEUTRAL_PROMPT` | `"a landscape"` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| `ALPHAS_C` | `[0]` |
| `ALPHAS_T` | `[0, 2, 4, 6, 8]` |
| `ALPHAS_G` | `[0, 5, 10]` |

### Observations
- Steering visible — content-matched prompts confirmed working.
- Last column (`CLIP-G=10`) destroys images on all rows — ceiling is between 5 and 10.
- `CLIP-G=5` functional; finer resolution needed in this range.

---

## Experiment #5 — Finer CLIP-G sweep below destruction threshold

**Date:** 2026-04-22

### Config changes from #4
| Parameter | Value |
|---|---|
| `ALPHAS_G` | `[0, 2, 4, 6, 8]` (was `[0, 5, 10]`) |

**Total combinations: 5 × 5 = 25**

### Observations
- Last column (`CLIP-G=8`) working but steering to a different art style, not realistic.
- CLIP-G hook was missing `clip_negative` — doing unconstrained projection removal in both directions.
- Fixed: added `clip_negative` to CLIP-G hook.

---

## Experiment #6 — T5 only (CLIP-L=0, CLIP-G=0)

**Date:** 2026-04-22

### Goal
Isolate T5's contribution to style removal. Determine whether T5 alone can steer toward realism before adding CLIP-G back in.

### Config
| Parameter | Value |
|---|---|
| `ALPHAS_C` | `[0]` |
| `ALPHAS_T` | `[0, 2, 4, 6, 8, 10, 12]` |
| `ALPHAS_G` | `[0]` |

### Observations
- TBD

---

---

## Experiment #7 — Multi-anchor neutral prompt

**Date:** 2026-04-27
**Author:** Tarun (fork: riptide-06/DiT-steering-vector, branch: tarun/style-unlearning)

### Hypothesis
Exp #6 showed T5-only steering successfully removes watercolor but lands in oil-painting territory rather than photorealism. The diagnosis: the direction `concept − single_neutral` doesn't point toward "no style," it points toward whatever is mathematically opposite of watercolor in T5 embedding space — which appears to be oil painting.

Fix: replace the single neutral with the **mean of multiple non-watercolor anchor prompts**. This should isolate the watercolor component without aiming the direction at any specific competing style.

### Config
| Parameter | Value |
|---|---|
| `CONCEPT_PROMPT` | `"a landscape in watercolor style"` |
| `NEUTRAL_PROMPTS` | `["a landscape", "a photograph of a landscape", "a landscape in oil painting style", "a landscape, pencil sketch", "a landscape, digital art"]` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| `NUM_STEPS` | `28` |
| `GUIDANCE_SCALE` | `7.0` |
| `SEED` | `42` |

Neutral activation = mean across all 5 neutral prompt captures (per step for context, single vector for pooled).

### #7a — Broad sweep
| Parameter | Values |
|---|---|
| `ALPHAS_C` | `[0]` |
| `ALPHAS_T` | `[0, 100, 300, 600]` |
| `ALPHAS_G` | `[0, 2, 4, 6]` |

**Observations**
- Cells with `t≥300, g≥4` go off-manifold — magenta/yellow color blobs, not images.
- Middle cells (around `t=300, g=2`) look like coherent realistic-ish landscapes.
- No oil-painting drift — the failure mode from Exp #6 is gone.
- CLIP-G alone (top row, t=0) shifts the image meaningfully — it is doing real work, not just clipped to zero.

### #7b — Refined sweep in working region
| Parameter | Values |
|---|---|
| `ALPHAS_C` | `[0]` |
| `ALPHAS_T` | `[100, 200, 300, 400]` |
| `ALPHAS_G` | `[0, 1, 2, 3]` |

**Observations**
- All 16 cells stay on-manifold — no destruction.
- Watercolor washes are gone across the whole grid.
- Best-looking cells: `t=200 g=2`, `t=300 g=0`, `t=300 g=1` — coherent landscapes with real depth and atmospheric perspective.
- Outputs are still painterly (digital-painting-like), not photographs. SD3's prior on `"a landscape"` appears to be itself painterly, so direction subtraction alone may not reach photorealism.

### Key takeaways
- **Multi-anchor neutral fixes the oil-painting drift.** Averaging 5 non-watercolor anchors removes the watercolor component without biasing the direction toward any specific other style.
- **Working region for this technique**: roughly `t ∈ [100, 400]`, `g ∈ [0, 3]`, with `c = 0`. Beyond `g=3` paired with `t≥300` images go off-manifold.
- **Limitation**: outputs are non-watercolor but still stylized. Reaching true photorealism likely needs a positive "photograph" target rather than just subtracting watercolor, or modifying GEN_PROMPT itself.

### Next steps
- Try **adding** a positive direction toward `"photograph of a landscape"` rather than only subtracting watercolor.
- Test with `GEN_PROMPT = "a landscape"` (no style word) to separate prior-painterly-ness from explicit-prompt-painterly-ness.
- Consider running on UnlearnCanvas benchmark with multi-anchor approach to see if the gain generalizes across style-content pairs.

### Artifacts
- Code: `sd3/sd3_style_unlearning.ipynb` on branch `tarun/style-unlearning` of fork `riptide-06/DiT-steering-vector`
- Result grids: `sd3/style_unlearning_results/exp7a_multi_anchor_broad.png`, `exp7b_multi_anchor_refined.png`

---

---

## Experiment #8 — Additive "photograph" direction (negative result)

**Date:** 2026-04-27
**Author:** Tarun (fork: riptide-06/DiT-steering-vector, branch: tarun/style-unlearning)

### Hypothesis
Exp #7 cleanly removes watercolor but lands in painterly territory, not photorealism. Hypothesis: instead of subtracting watercolor, *add* a positive direction toward "photograph." Direction = `photograph − watercolor`, applied with `+= α * proj * d`.

### Config
| Parameter | Value |
|---|---|
| `TARGET_PROMPT` | `"a photograph of a landscape"` |
| `CONCEPT_PROMPT` | `"a landscape in watercolor style"` |
| `GEN_PROMPT` | `"a landscape in watercolor style"` |
| Direction | `target − concept`, normalised per encoder |
| Hook op | `+=` instead of `-=` on T5 and CLIP-G |

Two runs:
- **#8a** unclamped T5 projection. `ALPHAS_T = [0, 50, 150, 400]`, `ALPHAS_G = [0, 1, 3, 6]`.
- **#8b** clamped T5 projection (`min=0`). `ALPHAS_T = [0, 20, 50, 100]`, `ALPHAS_G = [0, 2, 5, 10]`.

### Observations
- **#8a**: low T5 alpha (50) shifted output to autumn/oil-painting. T5 ≥ 150 destroyed the image into abstract texture. CLIP-G additive had near-zero effect across the row — base pooled embedding has small projection onto the photograph direction, so additive kick is negligible.
- **#8b**: clamping to `min=0` made the intervention a no-op — every cell identical to baseline watercolor. The watercolor activation projects *negatively* onto the `photograph − watercolor` direction (it points away from photograph), so clamping zeros out all projections.

### Diagnosis
The `photograph − watercolor` direction in T5 space does not point toward "photograph" in any useful sense. Pushing along it yields oil-painting drift (same failure mode as Exp #2/#6 with subtractive steering) before destroying the image. T5 in SD3 does not appear to represent photo-vs-painted as a cleanly extractable linear axis from this prompt pair.

### Key takeaway
Additive linear steering toward "photograph" via simple prompt-difference directions **does not reach photorealism** on SD3 with this technique. Photorealism likely requires a different intervention: a different layer (e.g. attention projections, MM-DiT joint blocks rather than encoder embeddings), a learned probe direction, or modifying GEN_PROMPT itself rather than steering activations.

### Status
Negative result — closing this branch of investigation. Recommended next direction: drop GEN_PROMPT's "in watercolor style" tag and steer to suppress residual watercolor association on the bare prompt, OR try MM-DiT block-level interventions as in `experiments/sd3.5/`.

---
