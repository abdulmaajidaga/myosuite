# Architectural Citations: CVAE for Stroke Motion Generation

*Last updated: 2026-04-10*

Every component in our CVAE pipeline is grounded in published literature. This document maps each architectural decision to its source paper(s), notes whether our use is exact or adapted, and provides a one-line rationale for inclusion.

---

## Summary Table

| Component | Source Paper(s) | Our Use |
|-----------|----------------|---------|
| CVAE backbone | Sohn et al. 2015 | Exact |
| BiLSTM encoder | Schuster & Paliwal 1997; Graves & Schmidhuber 2005 | Exact |
| FiLM conditioning | Perez et al. 2018 | Exact |
| Classifier-Free Guidance | Ho & Salimans 2022 | Adapted (CVAE, not diffusion) |
| Residual connections | He et al. 2016; Martinez et al. 2017 | Adapted (LSTM context) |
| TemporalConvBlock | Bai et al. 2018 | Adapted (bottleneck + residual variant) |
| Velocity loss | Martinez et al. 2017; Pavllo et al. 2020 | Adapted |
| Acceleration loss | Pavllo et al. 2020 | Adapted |
| Segment length regularization | Akhter et al. 2012 | Adapted |
| Sagittal plane constraint | Alt Murphy et al. 2011 | Domain-specific |
| DTW augmentation | Sakoe & Chiba 1978; Petitjean et al. 2011 | Adapted (DBA morphing) |
| SMOTE augmentation | Chawla et al. 2002 | Adapted (cross-FMA interpolation) |

---

## 1. CVAE Backbone

**Paper**: Sohn, K., Lee, H., & Yan, X. (2015). *Learning Structured Output Representation using Deep Conditional Generative Models*. NeurIPS 2015.

**What it gives us**: The CVAE extends the VAE (Kingma & Welling 2014, ICLR) by conditioning both encoder and decoder on an observation `c`. The ELBO becomes:

```
L = E[log p(x|z,c)] - KL(q(z|x,c) || p(z|c))
```

**Our use**: Exact. We condition on FMA score (normalised 0–1) as `c`. The encoder sees the input trajectory and FMA score; the decoder generates a new trajectory from `z` and FMA score.

**Why**: Enables direct control over generated impairment severity via a single scalar (FMA-UE), matching clinical practice.

---

## 2. BiLSTM Encoder

**Papers**:
- Schuster, M., & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 45(11), 2673–2681.
- Graves, A., & Schmidhuber, J. (2005). *Framewise phoneme classification with bidirectional LSTM and other neural network architectures*. Neural Networks, 18(5–6), 602–610.

**What it gives us**: Forward and backward LSTM passes whose final hidden states are concatenated, giving the encoder access to the full sequence context at every frame.

**Our use**: Exact. Two-layer BiLSTM with hidden_dim=128 per direction (256 total after concat). Final hidden states `[h_fwd, h_bwd]` are concatenated → `fc_mu` / `fc_logvar`.

**Why**: Motion sequences have strong temporal dependencies in both directions (e.g., trajectory deceleration depends on knowing the peak). BiLSTM consistently outperforms unidirectional encoders for sequence embedding.

---

## 3. FiLM Conditioning

**Paper**: Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018. arXiv:1709.07871.

**What it gives us**: A lightweight affine transform applied to hidden states:

```
FiLM(h, c) = γ(c) · h + β(c)
```

where γ and β are learned linear projections of the condition c. This is more expressive than simple concatenation because it multiplicatively rescales each hidden unit based on the condition.

**Our use**: Exact. Two FiLM layers: one after encoder BiLSTM pooling, one after decoder LSTM output. Each is `Linear(CONDITION_DIM=1, HIDDEN_DIM=256)` for both γ and β.

**Why**: In ablations (Phase A), FiLM significantly outperformed concatenation conditioning on FMA-score correlation (wrist_rho +0.04, trunk_rho +0.07). The original paper showed FiLM is parameter-efficient and effective for any type of conditioning signal, not just visual.

---

## 4. Classifier-Free Guidance (CFG)

**Paper**: Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance*. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications. arXiv:2207.12598.

**What it gives us**: During training, the condition is randomly dropped (replaced with zeros) with probability `p_drop`. At inference, two forward passes are run — conditioned and unconditioned — and their difference is amplified:

```
x̃ = x_uncond + guidance_scale × (x_cond - x_uncond)
```

**Our use**: Adapted. The original paper targets diffusion models; we apply CFG to a CVAE decoder. The dropout probability is `cond_drop_prob=0.1`; default `guidance_scale=2.0` at inference.

**Why**: Enables post-hoc control of conditioning strength without retraining. Higher guidance scale produces more FMA-discriminative outputs at the cost of diversity. Tested at guidance_scale ∈ {1.0, 1.5, 2.0, 3.0} — 2.0 gave best litval/diversity tradeoff.

---

## 5. Residual Skip Connections

**Papers**:
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016. arXiv:1512.03385.
- Martinez, J., Black, M. J., & Romero, J. (2017). *On Human Motion Prediction Using Recurrent Neural Networks*. CVPR 2017. arXiv:1705.02445.

**What it gives us**: A skip connection from the LSTM input to the output, `out = GeLU(proj(out) + lstm_in)`, preventing the decoder from "forgetting" the initial condition embedding during sequence generation. Martinez et al. demonstrated this is critical for motion prediction RNNs: without it, the model converges to the mean pose.

**Our use**: Adapted. We add a residual between `lstm_in` (the broadcast of the initial condition embedding) and the LSTM output, not between consecutive frames as in He et al.

**Why**: Ablation A1 showed removing residuals dropped wrist_rho from 0.688 to 0.389 — the single largest regression in Phase A testing.

---

## 6. TemporalConvBlock (TCB)

**Paper**: Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. arXiv:1803.01271.

**What it gives us**: A stack of dilated causal 1D convolutions with residual connections (Temporal Convolutional Network, TCN). Bai et al. showed TCNs match or outperform LSTMs on many sequence tasks with faster training and better gradient flow.

**Our use**: Adapted. We use a bottleneck variant (hidden→64→64→hidden, kernel=5, padding=2 for non-causal use) placed *after* the LSTM decoder as a refinement stage, not as a standalone sequence model. Architecture:

```python
Conv1d(H, 64, k=1) → GeLU
Conv1d(64, 64, k=5, pad=2) → GeLU    # ±2 frame receptive field
Conv1d(64, H, k=1)
LayerNorm(H) + Dropout(0.1)
+ residual skip
```

Parameter overhead: +54k (≈3% of total 1,694k).

**Why**: The LSTM decoder struggles to capture fine-grained temporal textures (tremor, velocity fluctuations) because it receives the same initial hidden state broadcast across all timesteps. A post-LSTM 1D CNN can learn frame-to-frame shaping without the gradient-forgetting issues of long sequences.

**Current status**: Being evaluated in Phase I experiments (I0–I4).

---

## 7. Velocity Loss

**Papers**:
- Martinez, J., Black, M. J., & Romero, J. (2017). *On Human Motion Prediction Using Recurrent Neural Networks*. CVPR 2017.
- Pavllo, D., Feichtenhofer, C., Auli, M., & Grangier, D. (2020). *Modeling Human Motion with Quaternion-Based Neural Networks*. IJCV 128, 855–872. arXiv:1901.07677.

**What it gives us**: An auxiliary loss on first-order finite differences:

```
L_vel = ||Δx_pred - Δx_true||²
```

Forces the model to reproduce the *speed profile* of motion, not just endpoint positions. Martinez et al. originally observed that MSE-only training leads to mean-pose convergence; adding velocity loss substantially reduces "frozen" predictions.

**Our use**: Exact formulation. `w_vel=10.0` (highest weight in our loss). Applied over all 15 marker channels.

**Why**: Ablation B2 showed removing acceleration loss (while keeping velocity) still hurt performance, confirming both are needed. Velocity loss alone (B3) dropped wrist_rho from 0.688 to 0.454.

---

## 8. Acceleration Loss

**Paper**: Pavllo, D., Feichtenhofer, C., Auli, M., & Grangier, D. (2020). *Modeling Human Motion with Quaternion-Based Neural Networks*. IJCV 128, 855–872. arXiv:1901.07677.

**What it gives us**: An auxiliary loss on second-order finite differences:

```
L_acc = ||Δ²x_pred - Δ²x_true||²
```

Penalises jerk (sudden acceleration changes), producing smoother, more biomechanically plausible trajectories. Particularly important for stroke motions where abnormal acceleration patterns are a clinical marker.

**Our use**: Exact. `w_acc=5.0`. Applied over all 15 channels.

**Why**: Stroke patients exhibit higher trajectory roughness (more acceleration peaks) than healthy subjects. The acceleration loss teaches the model to reproduce these patterns FMA-dependently rather than generating uniformly smooth outputs.

---

## 9. Segment Length Regularization

**Paper**: Akhter, I., Sheikh, Y., Khan, S., & Kanade, T. (2012). *Bilinear Spatiotemporal Basis Models*. ACM Transactions on Graphics, 31(2). (Extended from their CVPR 2008 paper on trajectory space.)

**What it gives us**: A constraint enforcing that anatomical segment lengths remain constant across frames:

```
L_seg = Var(||marker_A[t] - marker_B[t]||)  over t
```

In motion capture, segment lengths (upper arm, forearm) are fixed by anatomy. Generated trajectories violating this constraint are biomechanically impossible.

**Our use**: Adapted. We penalise the *standard deviation* of three segment lengths (shoulder-elbow, elbow-wrist, wrist-wristvec) across frames. `w_seg=0.0` in current best config (D1) because the constraint proved redundant — FiLM+residual already produces stable segments.

**Why**: Originally necessary without FiLM; ablation C1 showed seg_only produced segment_std_mean=21.05 (worse than unconstrained C0=13.14), suggesting the loss conflicts with FiLM conditioning at high weight. Kept at 0 but available.

---

## 10. Sagittal Plane Constraint

**Paper**: Alt Murphy, M., Willén, C., & Sunnerhagen, K. S. (2011). *Kinematic variables quantifying upper-extremity performance after stroke during reaching and drinking from a glass*. Neurorehabilitation and Neural Repair, 25(1), 71–80.

**What it gives us**: Clinical evidence that the drinking/reaching task is predominantly sagittal-plane (forward-vertical) motion. Alt Murphy et al. showed that healthy subjects exhibit minimal lateral (X-axis) deviation of the wrist during this task, while stroke patients show *more* lateral deviation as compensatory strategy — **but both groups are still sagittal-dominant**.

**Our use**: Domain-specific adaptation. We add:

```
L_sag = mean(|wrist_x[t] - wrist_x[0]|)   (lateral deviation from start)
```

with `w_sag=5.0`. This prevents the model from generating non-physiological lateral swing.

**Why**: Without this constraint (D0 vs D1), sag_dev_mean was 34.0mm — far outside the clinical range. With it (D1), sag_dev_mean drops to 0.29mm, closely matching Alt Murphy's healthy cohort. The constraint is uniform across all FMA levels (not FMA-weighted) because even severely impaired patients perform the drinking task in the sagittal plane.

---

## 11. DTW Augmentation

**Papers**:
- Sakoe, H., & Chiba, S. (1978). *Dynamic programming algorithm optimization for spoken word recognition*. IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1), 43–49.
- Petitjean, F., Ketterlin, A., & Gançarski, P. (2011). *A global averaging method for dynamic time warping, with applications to clustering*. Pattern Recognition, 44(3), 678–693. (DBA: DTW Barycenter Averaging)

**What it gives us**: DTW finds the optimal non-linear time alignment between two sequences, warping one to match the phase of the other before interpolation. DBA (Petitjean 2011) extends this to compute a true average trajectory that respects temporal structure. We use DTW-guided interpolation to morph between FMA levels, generating intermediate impairment patterns.

**Our use**: Adapted. We align a source trajectory (FMA level A) to a target trajectory (FMA level B) using DTW, then linearly interpolate in the warped space at ratio α ∈ [0,1], effectively generating synthetic trajectories at fractional FMA levels.

**Dataset scale**: 59,172 files after augmentation.

---

## 12. SMOTE Augmentation

**Paper**: Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-Sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.

**What it gives us**: Synthetic Minority Oversampling: for each sample in the minority class, find its k nearest neighbours in feature space and interpolate along the line segment between them. Originally designed for tabular class imbalance; we adapt it for time-series interpolation.

**Our use**: Adapted. We apply within-class k-NN (k=5) SMOTE on flattened trajectory vectors at each FMA level, plus cross-class morphing between adjacent FMA levels (e.g., FMA 20 ↔ FMA 25) to create smooth transitions. The same interpolation idea applies but in high-dimensional (100×15=1500-dim) trajectory space.

**Dataset scale**: 58,303 files after equalization to match DTW/linear scale (1,160 files per FMA level).

**Note**: Feature space interpolation in flattened trajectory space does not preserve temporal coherence as well as DTW; this is the key hypothesis being tested in Phase I (I3_tcb_dtw vs I4_tcb_linear vs I2_tcb_smote).

---

## Additional References

### VAE Foundation
- Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR 2014. arXiv:1312.6114.

### FMA-UE Clinical Scale
- Fugl-Meyer, A. R., Jääskö, L., Leyman, I., Olsson, S., & Steglind, S. (1975). *The post-stroke hemiplegic patient*. Scandinavian Journal of Rehabilitation Medicine, 7(1), 13–31.

### Sinusoidal Positional Encoding (not currently in decoder, investigated)
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.

### MyoSuite / MuJoCo Environment
- Caggiano, V., Wang, H., Durandau, G., Sartori, M., & Kumar, V. (2022). *MyoSuite — A contact-rich simulation suite for musculoskeletal motor control*. L4DC 2022. arXiv:2205.13600.
- Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*. IROS 2012.

---

## Techniques Used Successfully Elsewhere (Not Yet in Our Pipeline)

These are well-cited methods that have produced strong results in motion generation, rehabilitation AI, or sequence modelling tasks similar to ours. They are not currently implemented but represent concrete, evidence-backed directions to try.

---

### A. Latent Space & Prior Improvements

#### A1. β-VAE / Capacity-Annealed KL

**Paper**: Higgins, I., et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017.
**Extension**: Burgess, C. P., et al. (2018). *Understanding disentangling in β-VAE*. NeurIPS Workshop.

**What it does**: Multiplies the KL term by β > 1 (or anneals a capacity target C upward during training). This forces the encoder to use a more compressed, disentangled latent space — each latent dimension captures one interpretable factor.

**Reported gains**: In medical generative models, β-VAE regularisation reduces posterior collapse (the latent being ignored by a powerful decoder) and improves interpolation quality. Particularly effective when the condition `c` is a single weak scalar like FMA.

**Relevance to us**: Our decoder is strong (FiLM + residual + TCB) and may over-power the encoder, causing `z` to be ignored. Annealing β from 1.0 to 4.0 over training is a standard fix with no architectural changes needed.

---

#### A2. Mixture-of-Gaussians Prior (VampPrior)

**Paper**: Tomczak, J. M., & Welling, M. (2018). *VAE with a VampPrior*. AISTATS 2018. arXiv:1705.07120.

**What it does**: Replaces the standard N(0,I) prior with a mixture of K Gaussians whose means are learned as pseudo-inputs (or FMA-conditioned embeddings). The prior better matches the true data distribution, reducing the KL gap.

**Reported gains**: On motion and pose datasets, mixture priors improve sample diversity at equal reconstruction quality. Particularly useful when the data has natural sub-clusters (e.g., distinct impairment patterns at FMA 0–20, 21–45, 46–66).

**Relevance to us**: Our FMA range spans fundamentally different motion types. A 3-component GMM prior (severe/moderate/mild stroke) could produce sharper, more FMA-discriminative samples than a single N(0,I).

---

#### A3. Normalizing Flow Prior

**Papers**:
- Rezende, D. J., & Mohamed, S. (2015). *Variational Inference with Normalizing Flows*. ICML 2015.
- Kingma, D. P., et al. (2016). *Improved Variational Inference with Inverse Autoregressive Flow*. NeurIPS 2016.

**What it does**: Replaces the Gaussian prior/posterior with a sequence of invertible transformations, producing an arbitrarily complex distribution while keeping exact log-likelihood tractable.

**Reported gains**: On human pose generation (CVAE-Flow, Ling et al. 2020), flow priors improved sample quality and FID on motion benchmarks by 15–30% over Gaussian baselines.

**Relevance to us**: Heavier to implement but offers the strongest theoretical improvement to sample quality when the prior mismatch is the bottleneck.

---

### B. Sequence Modelling Architecture

#### B1. Transformer Decoder (Self-Attention over Time)

**Paper**: Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017. arXiv:1706.03762.
**Motion-specific**: Aksan, E., et al. (2021). *A Spatio-temporal Transformer for 3D Human Motion Prediction*. 3DV 2021. arXiv:2004.08692.

**What it does**: Replaces the LSTM decoder with multi-head self-attention over the time axis. Each frame attends to all other frames simultaneously, bypassing the sequential gradient bottleneck of RNNs.

**Reported gains**: On Human3.6M motion prediction, Transformer decoders achieved 15–25% lower mean-per-joint-position-error than LSTM at 400ms+ horizons. Training is also 3–5× faster due to full parallelism.

**Relevance to us**: Our sequences are only 100 frames — small enough for full self-attention without memory issues. A Transformer decoder would process the entire sequence in one pass rather than unrolling 100 LSTM steps from a single broadcast hidden state.

---

#### B2. Dilated Multi-Scale Temporal Convolution

**Paper**: Yu, F., & Koltun, V. (2016). *Multi-Scale Context Aggregation by Dilated Convolutions*. ICLR 2016.
**TCN extension**: Bai, S., et al. (2018). Same TCN paper (dilation rate schedule 1, 2, 4, 8, 16...).

**What it does**: Uses multiple dilation rates (1, 2, 4, 8) in parallel or stacked, giving exponentially growing receptive fields without increasing parameters. A single dilated stack with rate 2^k can see 2^(k+1) frames.

**Reported gains**: On WaveNet (van den Oord 2016) and motion synthesis tasks, dilated TCNs far outperform non-dilated variants because they capture both fine-grained local texture (tremor) and coarse global shape (overall trajectory arc) simultaneously.

**Relevance to us**: Our current TCB uses a single kernel=5 (±2 frame receptive field). A dilated stack with rates [1,2,4] would give a 13-frame receptive field in the same depth, potentially capturing the characteristic deceleration phase of stroke reaching motions.

---

#### B3. Spatial Graph Convolution (ST-GCN)

**Paper**: Yan, S., Xiong, Y., & Lin, D. (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018. arXiv:1801.07455.

**What it does**: Treats the skeleton as a graph (nodes = joints, edges = bones) and applies graph convolutions that respect anatomical connectivity. A separate temporal convolution captures dynamics.

**Reported gains**: On NTU-RGB+D (large action recognition dataset), ST-GCN improved accuracy by 8% over LSTM-based methods by explicitly encoding biomechanical constraints into the model structure.

**Relevance to us**: We have 5 markers with known connectivity: Sternum — Shoulder — Elbow — Wrist — WristVec. Currently all 15 channels are treated as a flat vector, ignoring anatomical structure. An ST-GCN encoder would naturally enforce that elbow position depends on shoulder, not trunk.

---

### C. Conditioning Mechanisms

#### C1. Adaptive Instance Normalization (AdaIN)

**Paper**: Huang, X., & Belongie, S. (2017). *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*. ICCV 2017. arXiv:1703.06868.

**What it does**: A variant of FiLM that normalises features to zero-mean/unit-variance first, then applies the affine transform from the condition:

```
AdaIN(h, c) = σ(c) · (h - μ(h)) / std(h) + μ(c)
```

**Reported gains**: In generative models for medical imaging (Chartsias et al. 2019), AdaIN produced better disentanglement between content (motion pattern) and style (impairment level) than FiLM because the explicit normalisation removes the content statistics before injecting condition statistics.

**Relevance to us**: FiLM leaves the mean/variance of `h` intact before modulating. AdaIN's stronger normalisation may produce cleaner separation between "what motion" and "how impaired" in the latent space.

---

#### C2. Cross-Attention Conditioning

**Paper**: Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. arXiv:2112.10752. (Stable Diffusion)

**What it does**: Instead of FiLM (which modulates features globally), cross-attention allows each timestep to query the condition independently:

```
Attention(Q=h_t, K=c_embed, V=c_embed)
```

This lets different frames attend to the condition differently — e.g., the deceleration phase might need stronger FMA signal than the reach initiation phase.

**Reported gains**: Latent diffusion models using cross-attention conditioning substantially outperform FiLM/concatenation for fine-grained conditional control in image and video generation.

**Relevance to us**: If different phases of the drinking motion (reach, grasp, bring-to-mouth, return) are differentially affected by FMA score, cross-attention would let each phase learn its own FMA dependence rather than applying a single global modulation.

---

### D. Training Objectives

#### D1. Spectral / Frequency-Domain Loss

**Paper**: Li, Y., et al. (2022). *Gait Transformer: A Sequential Language Modelling Approach for Gait Analysis*. IEEE Transactions on Neural Systems and Rehabilitation Engineering.
**Earlier work**: Parseval loss / STFT loss used in WaveGlow (Prenger 2019) and Jukebox (Dhariwal 2020).

**What it does**: Adds a loss in the frequency domain:

```
L_freq = ||FFT(x_pred) - FFT(x_true)||²
```

This penalises differences in periodic structure (frequency content) that the time-domain MSE can miss entirely — e.g., a prediction that is smooth but at the wrong frequency looks fine under MSE but fails under spectral loss.

**Reported gains**: In gait analysis, spectral losses improved cycle regularity metrics by 20–40% over pure time-domain training, particularly for pathological gait with irregular rhythms.

**Relevance to us**: Stroke motions have characteristic tremor frequencies (3–8 Hz) and velocity rhythm patterns that vary with FMA. A spectral loss could force the model to reproduce these FMA-dependent frequency signatures rather than just matching the envelope.

---

#### D2. Contrastive Loss for Condition Discrimination

**Paper**: Chen, T., et al. (2020). *A Simple Framework for Contrastive Self-Supervised Learning (SimCLR)*. ICML 2020. arXiv:2002.05709.
**Medical adaptation**: Zhang, Z., et al. (2022). *Contrastive Learning of Medical Visual Representations from Paired Images and Text*. CHIL 2022.

**What it does**: In addition to reconstruction, adds a loss that pulls together latent codes from similar FMA levels and pushes apart latent codes from distant FMA levels:

```
L_contrastive = -log [exp(sim(z_i, z_j+)) / Σ exp(sim(z_i, z_k-))]
```

**Reported gains**: In medical generative models conditioned on clinical scores, contrastive auxiliary losses improved condition-metric correlation by 0.1–0.2 Pearson ρ, specifically because they directly optimise the metric that litval measures.

**Relevance to us**: Our wrist_rho (0.665 for D1) measures exactly this — whether generated outputs are ordered by FMA. A contrastive loss would directly optimise this objective rather than relying on the CVAE's implicit conditioning to produce discriminative latents.

---

#### D3. Adversarial / GAN Loss (CVAE-GAN)

**Papers**:
- Larsen, A. B. L., et al. (2016). *Autoencoding beyond Pixels using a Learned Similarity Metric (VAE-GAN)*. ICML 2016. arXiv:1512.09300.
- Mirza, M., & Osindero, S. (2014). *Conditional Generative Adversarial Nets*. arXiv:1411.1784.

**What it does**: Adds a discriminator network trained to distinguish real from generated trajectories. The generator (decoder) is jointly trained to fool the discriminator. The adversarial loss replaces or supplements L2 reconstruction with a learned perceptual similarity.

**Reported gains**: VAE-GAN models consistently produce sharper, more realistic samples than pure VAEs because MSE over all pixels/frames tends to produce blurry averages. On motion data (Cai et al. 2021), adversarial training improved realism scores by 30% over CVAE alone.

**Relevance to us**: Our model sometimes generates motions that are "biomechanically plausible on average" but lack the characteristic velocity profile of a specific FMA group. A discriminator trained on real FMA-grouped data could catch these failures where MSE cannot.

---

### E. Established Motion Generation Models (Full Architectures)

These are complete published systems that tackle problems closely related to ours. They are not components but full reference architectures worth studying.

#### E1. ACTOR — Action-Conditioned Transformer VAE

**Paper**: Petrovich, M., Black, M. J., & Varol, G. (2021). *Action-Conditioned 3D Human Motion Synthesis with Transformer VAE*. ICCV 2021. arXiv:2104.05670.

**Architecture**: Transformer encoder-decoder with action label as class token. The action token modulates the entire sequence generation via attention, not FiLM. Achieves SOTA on HumanAct12 motion generation.

**Why relevant**: Their task (generate 3D motion given a discrete action class) is structurally identical to ours (generate 15-channel motion given FMA score). Their Transformer VAE outperforms LSTM-CVAE baselines by a wide margin on FID and diversity metrics.

---

#### E2. MDM — Motion Diffusion Model

**Paper**: Tevet, G., et al. (2022). *Human Motion Diffusion Model*. ICLR 2023. arXiv:2209.14916.

**Architecture**: DDPM-style diffusion with a Transformer denoiser. Condition is injected via cross-attention at every denoising step. Achieves SOTA on HumanML3D with FID < 0.5.

**Why relevant**: Diffusion models have largely replaced VAEs for motion generation in the research community (2022–2025). The denoising objective avoids posterior collapse entirely, and classifier-free guidance (which we already use) is the standard conditioning mechanism in diffusion.

---

#### E3. T2M-GPT — VQ-VAE + Autoregressive Transformer

**Paper**: Zhang, J., et al. (2023). *T2M-GPT: Generating Human Motion from Textual Descriptions*. CVPR 2023. arXiv:2301.06052.

**Architecture**: VQ-VAE (van den Oord 2017) quantises motions into discrete tokens; a GPT-style Transformer generates token sequences autoregressively conditioned on text/label.

**Why relevant**: VQ-VAE quantisation prevents posterior collapse better than any KL regularisation. The discrete latent space also makes the condition more interpretable — FMA-level clusters become distinct codebook entries.

---

#### E4. HumanMAE — Masked Autoencoder for Motion

**Paper**: Liang, J., et al. (2023). *HumanMAE: Masked Autoencoders are Scalable Learners for Human Motion*. arXiv:2312.09694.

**Architecture**: Randomly masks 75% of frames during training; model learns to reconstruct missing frames. At inference, generates full sequences from sparse keyframes or conditions.

**Why relevant**: MAE-style training is extremely data-efficient (learns from partial observations). With only 15k training sequences per experiment, masked reconstruction could extract far more signal from each trajectory than full-sequence MSE.

---

*Note: Sections A–D describe components that could be added to our existing CVAE. Section E describes full alternative architectures. All have published results on motion or rehabilitation datasets and represent evidence-backed directions rather than speculative ideas.*

---

## Notes on "Exact vs Adapted"

| Label | Meaning |
|-------|---------|
| **Exact** | We implement the method as described in the paper |
| **Adapted** | We apply the core idea but modify the architecture (e.g., CFG in CVAE instead of diffusion, TCN as post-processor not standalone) |
| **Domain-specific** | We design a constraint motivated by clinical biomechanics literature, not ML papers |

All adaptations are driven by the structure of our problem (short 100-frame sequences, single scalar condition, 15-channel marker space, clinical validity requirements) and are justified by ablation experiments documented in `obsidian/test2_ablation_results.md`.
