# Implementation and manuscript mapping

## Core mapping

| Manuscript element | Implementation |
|---|---|
| Matching/pair context encoders | `core/extractor.py`, `PIVFlow.fnet/cnet` |
| Initial displacement and convex weights | `init_head`, `mask_head` |
| Four-level all-pairs volume | `core/corr.py` |
| Motion encoder, recurrent proposal | `core/update.py` |
| Endpoint signatures and ordered quantiles | `core/verifier.py` |
| Bounded normalized signed gain | `utility_target` in `core/losses.py` |
| Retain/half/full selection and detached feedback | `select_step`, `PIVFlow.forward` |
| Two observations, four comparisons, eight evaluations | `replay_loss` |
| Committed/trial sequence supervision | `sequence_loss` |
| EMA, consistency mask and three training phases | `core/training.py` |
| Worker/conveyor fixed procedures | `applications/` |

## Architecture

- ResNet-18/34 BasicBlocks with 7x7 stride-two stem, stride-two max pooling,
  layer2 stride two, layer3/4 dilation two/four. A final 1x1 projection sets the
  matching width. Context projection outputs twice the recurrent width.
- Six-channel context stem repeats ImageNet input weights with a factor of 1/2.
  Running batch-normalization statistics are frozen; affine parameters train.
- Initial context is ReLU and recurrent state is tanh. Motion encoders and heads
  use ReLU. ConvGRU uses 3x3 gates. Hidden state continues after retention.
- Flow heads use 3x3 -> ReLU -> 1x1. Convex-mask logits are multiplied by 0.25.
- Verification features, endpoint adapters and context adapters have width 32;
  the head has two 64-channel 3x3 layers and a two-channel 1x1 output.
- Input normalization is ImageNet normalization **after** the RGB intervention.
- Only bottom/right replicate padding is used. The stride-eight label origin is
  (0,0). Image predictions are cropped; padded positions never acquire labels.
- Bilinear endpoint/correlation interpolation and gain calculations use float32
  under autocast. Outside-grid samples are zeroed and retain validity channels.
- Augmentation uses bilinear dense-flow resize, scaled vector components,
  nearest-neighbor validity, horizontal flips and shared spatial crops. This is
  distinct from stride-origin reference sampling for utility supervision.
- Full reference support is preserved during EPE evaluation. Retention is not
  pixel rejection, and nonfinite predictions cause an evaluation failure.

Use `tools.profile_model` to report parameter counts, latency and memory usage.

## Gradient boundaries

The additive incoming displacement detaches at every iteration. Hidden state
does not detach across iterations. Hard indices and the four-channel feedback
detach. Selection runs under `no_grad`. Replay recomputes the verifier with live
adapter/head parameters but detached matching features, context, correlations
and candidate coordinates. Verifier losses cannot update the proposal backbone.
The full-step trial loss trains proposals even when their selected scale is zero.

Replay supervises utility separately under each observation. There is no forced
cross-observation equality of features, quantiles or proposal trajectories.
Real Mixed-U pseudo-labels supervise only flow consistency, never utility targets.

## Controls

- Paired mean gain: two output channels fit the normalized utility mean with MSE;
  the first channel supplies ranking. This keeps the head dimensions fixed.
- Marginal quantile/mean error: predict anchor and trial errors conditioned on
  the same paired evidence; score anchor error minus trial error.
- Confidence control: binary positive-gain target with logit threshold zero.
- Three-action control: classify the oracle eligible zero/half/full action with
  zero as the reference logit and shared trial scorers. Its feedback is logit
  feedback, not a calibrated gain.
- Independent augmentation evaluates each origin comparison twice on its own
  observation to preserve the number of supervised comparisons.
- RGB removes the verifier module entirely. Its three-phase schedule includes
  an idle frozen-proposal interval corresponding to verifier initialization.
- The approximately parameter-matched RGB-S control adds a trainable residual
  1x1 motion adapter (128 -> 488 -> 128): 125,544 added parameters, compared
  with 125,698 parameters in the default S verifier. The difference is 154
  parameters; these are functional motion parameters rather than unused tensors.

Run external baselines with their official implementations and evaluate their
exported predictions using the common manifest.

## Checkpoints and random seeds

Save/load includes optimizer moments, GradScaler, phase/step, per-rank random
states and data cursor. Geometry augmentation is keyed by epoch and sample index;
the sampler is keyed by seed and epoch. Worker prefetch cannot change a sample's
geometry. Resume with the same world size, batch/worker settings and data.
Mixed-U records the EMA teacher.
