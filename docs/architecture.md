# Architecture and experiment boundaries

The baseline is Ultralytics 8.3.203 YOLOv8n with a four-class detection head. The custom
trainer reconstructs each chosen architecture before loading compatible pretrained weights.
This prevents the training engine from silently reverting to the baseline.

| Variant | Change to YOLOv8n |
|---|---|
| baseline | Standard YOLOv8n detection model |
| bot | Residual multi-head attention after backbone layer 8 |
| lska | Separable spatial attention after layers 9 and 21 |
| ghost | Replace neck C2f layers 12, 15, 18, 21 with C2fGhost |
| lscd | Shared GroupNorm head replacing the final Detect layer |
| pd | Combine all four modifications |

The backbone retains three output resolutions. The shared detection head projects each
scale to 64 channels, applies the same pair of 3x3 GroupNorm convolutions, and uses shared
classification/regression projections with learned per-scale regression multipliers.
Ultralytics supplies the existing detection loss, distributional box decoding, and NMS.

## Differences from an exact reproduction

- BOT uses depthwise convolution for positional context and residual multi-head attention
  after an existing C2f layer. This is not an exact reconstruction of the paper's BOT block.
- LSKA uses factorized depthwise kernels and a residual sigmoid gate. The paper's text,
  acronym usage, diagrams, and selective-kernel reference leave implementation ambiguity;
  its adaptive multi-branch kernel-selection operation is not recreated here.
- C2fGhost uses Ultralytics' GhostBottleneck implementation inside C2f.
- LSCDHead uses a fixed hidden width of 64 and a shared classification bias prior. The
  paper does not establish all implementation details required for an exact replica.
- The package's standard loss is retained. Claims in the paper about the baseline loss
  are not used to override the actual pinned implementation.
- New wrapper layers change some checkpoint key names. Compatible keys are transferred;
  the transfer count is printed during training. New or unmatched parameters start randomly.
- No paper accuracy, latency, or parameter counts are assumed. Measure this implementation
  on the prepared dataset and record the resulting values.

Use the baseline for the first project result. Label modified-model results **PD-inspired**
in reports until the authors' implementation can be obtained and compared. Embedded-device
deployment, damage severity, segmentation, and tracking are outside this implementation.
