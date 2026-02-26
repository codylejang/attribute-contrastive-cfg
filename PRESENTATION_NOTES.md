# Presentation Script

## Opening (~30s)

"In HW3 I built a CFG model on CelebA — 40 binary attributes, 64x64 faces. It worked, but at high guidance scales everything collapsed into the same face. I started asking: *why?* The answer is that CFG's null baseline is a blunt instrument — it captures everything that makes your condition different from nothing, including correlations you didn't ask for. So I tried replacing null with something smarter."

## Method (~45s)

"Standard CFG contrasts your target condition against zeros. The problem is that direction encodes *all* 40 attributes at once — Smiling, Male, Young, and every correlation between them.

My fix: instead of contrasting against nothing, contrast against a condition that's identical to your target except the one attribute you care about is flipped. So to isolate Smiling, I contrast [Smiling=1, Male=1, Young=1] against [Smiling=0, Male=1, Young=1]. Now the guidance direction captures *only* what Smiling changes.

One line of code. No retraining. Same two forward passes."

## Results (~60s)

"Three findings.

**First, KID.** This is the main result. As you crank up guidance scale from 1 to 7, standard CFG's quality degrades 151%. Contrastive only degrades 37%. At w=7 the gap is over 4 standard deviations — the error bars don't even come close to overlapping.

**Second, attribute leakage.** Standard CFG pushes Male detection to 100% at w=3 even though we only conditioned on Smiling. Contrastive holds at 97% across all scales. It's not amplifying correlated attributes the way standard CFG does.

**Third, LPIPS diversity.** At w=2, contrastive is 7.7% more diverse. I want to be honest here — at high w, LPIPS actually goes back *up* for both methods. That's artifacts inflating pixel distances, not real diversity. So I trust LPIPS only at moderate guidance scales.

At w=1, both methods give identical outputs — same KID, same LPIPS. So every difference you see at higher w comes purely from the guidance direction."

## Why It Works (~20s)

"The anchor is one bit flip away from the target. That means the guidance direction has a much smaller L2 norm. Same w, less extrapolation off the data manifold. That's why contrastive doesn't collapse at high w."

## Limitations (~20s)

"Two main ones. Our CelebA subset has zero examples for some attributes — Eyeglasses, Heavy_Makeup, Wearing_Hat. No guidance method fixes missing training data. And this relies on binary attributes where you can do exact bit flips — extending to free-form text requires approximate prompt engineering."

## Takeaway (~15s)

"The baseline in CFG is a design choice, not a given. Replacing null with a targeted anchor gives you 45% better KID at w=7, reduces attribute leakage, and costs nothing. It's a drop-in improvement for any pretrained CFG model."

---

## Q&A Cheat Sheet

**"Isn't this just one line of code?"**
"Yes — that's the point. The insight is *why* null is a bad baseline and *what* to replace it with. We validated it across three metrics and five guidance scales."

**"KID error bars overlap at w=2."**
"At w=2 the improvement is modest. The real story is the sweep — at w=7 it's >4σ, and the gap grows monotonically. Standard degrades 151%, contrastive only 37%."

**"Why not retrain?"**
"Wanted to isolate the effect of guidance direction from architecture. No retraining means this works on any pretrained CFG model."

**"Contrastive prompts paper (Wu & De la Torre)?"**
"Same core idea, different setting. They use approximate text prompts in CLIP space. We have binary attributes — exact single-bit flips, clean controlled experiments, quantitative evaluation they didn't do."

**"CFG++ / Dynamic Negative Guidance?"**
"They fix *how much* to extrapolate. We fix *which direction*. Complementary — you could combine them."

**"Mode collapse at high w — why?"**
"The null-to-conditional direction has large L2 norm. Multiplying by w=7 pushes way off manifold. Contrastive direction has smaller norm — same w, less damage."

**"LPIPS goes up at high w?"**
"Off-manifold artifacts inflate pixel distances. It's not real diversity — the faces look collapsed but the noise patterns differ. We only trust LPIPS at moderate w."

---

## Numbers to Know

| Metric | Standard | Contrastive | Gap |
|---|---|---|---|
| KID (w=2) | 0.0364 | 0.0330 | −9.4% |
| KID (w=7) | 0.0777 | 0.0424 | −45%, >4σ |
| KID degrade w=1→7 | +151% | +37% | — |
| LPIPS (w=2) | 0.182 | 0.196 | +7.7% |
| Male % (w=2) | 99.7% | 96.7% | less leakage |
| Cost | 2 fwd passes | 2 fwd passes | zero |
