# STT-MRAM five-curve reproduction specification

## Source and goal

The repository reproduces the 7/9-rate sparse-code STT-MRAM study and extends
it with BCH and Deep-FFNN branches. The current task adds a controlled
BER-versus-write-error-rate (`P1`) experiment. Figure 6 is used only to select
the horizontal sweep; its plotted BER values are not copied.

## Controlled P1 experiment

- Hypothesis: increasing `P1` should not improve BER when all other channel,
  code, decoder, model, and Monte Carlo settings are fixed.
- Controlled variable: `P1` in `2e-8, 2e-7, 2e-6, 2e-5, 2e-4, 2e-3, 2e-2`.
- Fixed variable: `sigma_mu = 10%` (`sigma_ratio = 0.10`).
- Metrics: payload BER and FER.
- Comparison curves: the five implementations already used by
  `all_curves_sigma10_15.csv`: `SLNN [1]`, `without coding`, `BCH+sparse`,
  `only-BCH`, and `only-sparse (ML decoding)`.
- Output: `results/all_curves_p1_sigma10.csv`, with paired BER/FER columns for
  every included method.
- The P1 runner can select another fixed sigma and can omit `only-BCH` for a
  faster diagnostic sweep; filenames record the selected sigma.
- Reproducibility: seed 42, the existing global FFNN checkpoint, incremental
  CSV writes, and a dedicated JSON manifest. Common random-number streams are
  intentionally reused across P1 points to reduce comparison variance.

## Existing end-to-end branches

1. `SLNN [1]`: 7 payload bits -> Table-1 sparse encoder (7 to 9) -> channel ->
   global Deep FFNN -> 7 decoded bits.
2. `without coding`: 7 raw bits -> channel -> threshold detector; BER is
   evaluated analytically.
3. `BCH+sparse`: 16 payload bits -> BCH(31,16,7) -> four shaping bits -> five
   sparse blocks -> channel -> FFNN -> BCH decoder -> 16 decoded bits.
4. `only-BCH`: 16 payload bits -> BCH(31,16,7) -> channel -> threshold detector
   -> bounded-distance BCH decoder.
5. `only-sparse (ML decoding)`: 7 payload bits -> sparse encoder -> channel ->
   alpha=2.5 Euclidean lookup decoder.

## Channel assumptions

- `mu0 = 1 kOhm`, `mu1 = 2 kOhm`, no offset.
- `P0 = P1 / 100`, `Pr = P1 / 100`, read direction `write0`.
- Equal relative resistance spread for states 0 and 1.

## Verification checklist

- The P1 table has exactly seven ordered rows and paired BER/FER columns per
  included curve.
- Every row has `sigma_mu = 10`.
- The original sigma-sweep outputs remain untouched by P1-only execution.
- All BER values are finite and lie in `[0, 1]`.
- The checkpoint hash and resolved experiment settings are recorded.

## Ambiguities and assumptions

- The supplied table includes `2e-2`, although the example Figure 6 image ends
  at `1e-3`; the explicit user-provided table is treated as authoritative.
- Monte Carlo branches stop at the existing bit-error target or maximum-frame
  cap, so very low BER estimates may be upper-resolution-limited; confidence
  information remains available in the simulation records but the requested
  compact CSV contains both BER and FER.

## Classical Euclidean-versus-Mahalanobis experiment

This is a user-requested repository extension, not a result claimed by either
supplied paper. Both compared decoders are analytical and use no trained model.

### Hypothesis and controlled comparison

- Hypothesis: pooled-covariance Mahalanobis distance reduces sparse-decoder BER
  and FER relative to pure Euclidean distance when cell-position variances are
  unequal.
- Baseline: `argmin_c (r-c)^T(r-c)` over all 128 sparse-code centroids.
- Proposed metric: `argmin_c (r-c)^T Sigma^-1(r-c)` over the same centroids.
- Shared controls: payloads, encoded codewords, channel samples, analytical
  centroids, seed 42, batch size, stopping rules, and sigma sweep.
- Metrics: 7-bit payload BER and FER with Wilson 95% confidence intervals.
- Output: `results/classical_distance_comparison.csv` and
  `results/run_manifest_classical_distance.json`.

### Centroids and covariance mapping

- The centroid for target bit `x` is the exact conditional mean `E[R|x]` of
  the implemented BAC/read-disturb/Gaussian-mixture channel.
- `Sigma` is the analytical pooled within-class residual covariance across all
  128 equiprobable sparse classes.
- The channel implementation samples cells conditionally independently, so
  off-diagonal covariance entries are zero. Diagonal entries still differ by
  position because the Table-1 codebook has unequal one fractions across its
  nine cells and state 0/state 1 have unequal resistance variances.
- No alpha attenuation, FFNN, fitted parameter, or training data is used.

### Verification checklist

- The covariance is positive definite and diagonal under the implemented
  independent-cell channel.
- It is not a scalar multiple of identity at the tested channel settings;
  otherwise Mahalanobis and Euclidean rankings would be identical.
- Both decoders recover all 128 classes exactly at their noiseless analytical
  centroids.
- Both decoders consume exactly the same resistance samples in every batch.

### Known ambiguity

The supplied screenshot does not specify how `Sigma` is estimated. This
experiment uses the pooled within-class covariance, the standard shared-
covariance nearest-centroid interpretation, and records that choice explicitly
instead of silently fitting covariance from evaluation data.

### Controlled comparison result

The paired seed-42 sweep used 200,000 frames at every sigma point, `P1=2e-4`,
no offset, and the same resistance samples for both metrics.

| sigma_mu | Euclidean BER | Mahalanobis BER | relative Mahalanobis change |
|---:|---:|---:|---:|
| 10% | 5.492857e-4 | 5.642857e-4 | +2.73% |
| 11% | 1.519286e-3 | 1.525000e-3 | +0.38% |
| 12% | 3.377143e-3 | 3.426429e-3 | +1.46% |
| 13% | 6.439286e-3 | 6.635714e-3 | +3.05% |
| 14% | 1.143000e-2 | 1.160500e-2 | +1.53% |
| 15% | 1.858786e-2 | 1.883357e-2 | +1.32% |

Across all six points, Euclidean produced 58,664 bit errors and Mahalanobis
produced 59,626 on 8.4 million payload-bit trials per decoder. Thus this exact
pooled-covariance substitution did not improve the decoder; its aggregate BER
was about 1.64% higher. Per-point Wilson intervals overlap substantially, so
the defensible conclusion is “no observed benefit,” not a claim of a large or
universal degradation.

## BCH(15,7) + sparse joint-decoder experiment

This is a repository extension requested by the user, not a method claimed by
either supplied paper.

### Hypothesis and controlled comparison

- Hypothesis: using all 27 analog resistance observations in one exact
  maximum-likelihood decision over the 128 valid messages reduces payload BER
  and FER relative to the cascaded decoder on the same received frames.
- Baseline: three independent global-FFNN sparse decisions, followed by hard
  7-bit blocks and a bounded-distance BCH(15,7,5) decoder (`t=2`).
- Proposed decoder: `m_hat = argmax_m P(y_1:27 | m)` over all 128 messages.
- Controlled variables: transmitted messages, sparse mapping, channel samples,
  channel parameters, random seed, batch size, and stopping rules are shared.
- Metrics: 7-bit payload BER and FER, plus evaluation time per decoder.

### End-to-end data flow and shapes

1. Payload `[N,7]` -> systematic BCH(15,7,5) -> `[N,15]`.
2. Append six fixed zero padding bits -> `[N,21]`.
3. Reshape into three parallel 7-bit blocks -> `[N,3,7]`.
4. Apply the exact Table-1 sparse LUT independently -> `[N,3,9]`.
5. Flatten physical cells -> channel -> resistance tensor `[N,27]`.
6. Baseline: FFNN on `[3N,9]`, hard 7-bit outputs, remove six padding
   positions, then bounded-distance BCH decoding.
7. Joint ML: precompute the 128 valid physical codewords `[128,27]`; evaluate
   the channel log-likelihood of every candidate and return its 7-bit index.

### Likelihood mapping

For a target physical bit `x`, the combined write/read-disturb channel first
produces hidden state `s`, then the resistance follows the state-dependent
Gaussian. Therefore each cell likelihood is a two-component Gaussian mixture:

- `p(y|x=0) = q0*N(y;mu0,sigma0) + p0*N(y;mu1_eff,sigma1_eff)`
- `p(y|x=1) = p1*N(y;mu0,sigma0) + q1*N(y;mu1_eff,sigma1_eff)`

Cell outputs are assumed conditionally independent given the candidate. Scores
are accumulated in log space. This exactly matches `paper79.channel` and does
not use the FFNN, hard bits, Euclidean attenuation, or an independence
approximation between BCH/sparse decisions.

### Verification checklist

- BCH codebook has shape `[128,15]`, is systematic, and has minimum distance 5.
- All error patterns of weight 0, 1, or 2 decode to the transmitted payload.
- Padding positions 15..20 are zero for all candidates.
- Candidate physical codebook has shape `[128,27]` and 128 unique rows.
- At resistance means with negligible channel noise, all 128 messages decode
  exactly under joint ML.
- Baseline and joint decoders consume identical resistance samples in the
  controlled experiment.

### Known ambiguity

- “padding 6 bit trống” is interpreted as six fixed zero bits appended after
  the 15-bit BCH word. They carry no information and are included in all 27
  likelihood observations because their sparse encodings still depend on the
  surrounding 7-bit block values.

### Controlled sigma-sweep result

The joint experiment was run with the same evaluation conditions as
`results/all_curves_sigma10_15.csv`: `P1=2e-4`, sigma 10..15%, no offset,
batch size 5,000, seed 42, a 500-bit-error target, and a 5,000,000-frame cap.
The existing global FFNN checkpoint (SHA-256 beginning `c33e3e`) was reused.

| sigma_mu | sequential BER | joint ML BER | joint bit errors |
|---:|---:|---:|---:|
| 10% | 1.524857e-4 | 0 | 0 |
| 11% | 3.044857e-4 | 0 | 0 |
| 12% | 7.709143e-4 | 0 | 0 |
| 13% | 1.973286e-3 | 6.571429e-7 | 23 |
| 14% | 4.303257e-3 | 3.457143e-6 | 121 |
| 15% | 8.257229e-3 | 8.514286e-6 | 298 |

All six points reached the frame cap because joint ML did not accumulate 500
bit errors. A reported zero is therefore an empirical zero, not proof of zero
BER. With 35,000,000 payload-bit trials and zero observed errors, the recorded
95% Wilson upper bound is approximately `1.10e-7`.

An extended low-BER run at `sigma_mu=12%` raised the cap to 50,000,000 frames
and stopped at 43,365,000 frames after observing 21 joint-decoder bit errors
across six erroneous frames. The estimate is `BER=6.918021e-8` (95% Wilson CI
`[4.525000e-8, 1.057658e-7]`) and `FER=1.383604e-7` (95% Wilson CI
`[6.341096e-8, 3.018974e-7]`). This confirms that the zero in the standard
5,000,000-frame sweep was a finite-sampling result rather than a zero-error
implementation. Detailed output and resolved configuration are stored under
`results/joint_extended_sigma12/`.
