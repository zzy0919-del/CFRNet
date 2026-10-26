# CFRNet

PyTorch implementation of **Consistency-Conditioned Fading Rescue Network** for infrared small-target segmentation.

CFRNet combines an attention-based encoder-decoder, bidirectional semantic-frequency interaction with decoupled reliability routing, and stage-wise evidence preservation. Early target evidence is propagated through the encoder; the deep rescue branch writes back only when cross-stage evidence is consistent and the current response is fading.

## Repository

```text
CFRNet/
├── cfrnet/
│   ├── model.py       # Complete CFRNet architecture
│   ├── blocks.py      # Attention, frequency interaction and evidence rescue
│   ├── loss.py        # Multi-scale SLS loss
│   ├── data.py        # Dataset loading and augmentation
│   ├── metrics.py     # Pixel and object metrics
│   └── seed.py        # Reproducible random seeds
├── splits/            # Fixed train/test image lists
├── train.py
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.9 or newer. Install a matching PyTorch/torchvision pair for your machine, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data

Place datasets outside the tracked source files:

```text
dataset/
└── NUAA-SIRST/
    ├── images/
    │   ├── <image_id>.png
    │   └── ...
    └── masks/
        ├── <image_id>.png
        └── ...
```

Each line of `splits/<dataset>/train.txt` or `test.txt` identifies one image/mask pair. Supported datasets are `NUAA-SIRST`, `NUDT-SIRST` and `IRSTD-1k`.

Images are converted to grayscale and scaled to [0, 1]; nonzero mask pixels are foreground. Training uses horizontal/vertical flips and rotations within ±3°. Inputs are cropped/padded rather than resized. Target sizes are 256×256 for NUAA/NUDT and 512×512 for IRSTD-1k; evaluation uses a center crop for larger images.

## Training

Run commands from this directory:

```bash
python train.py --dataset-root ./dataset --dataset NUAA-SIRST --device cuda:0
```

The default run executes **1000 epochs**, without early stopping. Adam uses a cosine learning-rate schedule from 1e-3 to 1e-5 over 1000 epochs. Batch size is 8 for NUAA/NUDT and 4 for IRSTD-1k. Seed is 42. The first epoch supervises the finest decoder head; subsequent epochs use the fused prediction plus four auxiliary heads with the original SLS loss.

Evaluation starts at epoch 300, repeats every 5 epochs, and also runs at the final epoch. The best checkpoint is selected by pixel F1 at threshold 0.5. Following the supplied experiment, evaluation and checkpoint selection use the fixed **test split**; these selected scores are not an independent held-out validation estimate. Object detection uses centroid distance <3 pixels. No dataset images, trained weights or claimed benchmark scores are included.

```bash
# Another dataset
python train.py --dataset-root ./dataset --dataset IRSTD-1k --device cuda:0

# Resume with the same dataset, schedule and other training options
python train.py --dataset-root ./dataset --dataset NUAA-SIRST --device cuda:0 --resume runs/CFRNet/last.pth

# Short CPU check using real prepared data
python train.py --dataset-root ./dataset --epochs 1 --eval-start 1 --workers 0 --batch-size 2 --device cpu --output runs/smoke
```

Outputs in `runs/CFRNet/`: `config.json`, `history.csv`, `last.pth` (every epoch, resumable) and `best.pth` (best evaluated F1). Resume requires the original epoch horizon and training options; `--epochs` is the total horizon, not additional epochs.

## Model usage

```python
import torch
from cfrnet import CFRNet

model = CFRNet().eval()
with torch.no_grad():
    logits = model(torch.randn(1, 1, 256, 256))
    probabilities = logits.sigmoid()
```

Input height and width must be at least 32 and divisible by 16. `model(x, return_aux=True)` returns `pred` and the four native-scale `aux` logits. For trained inference, load `checkpoint["model"]` from `best.pth` with `model.load_state_dict(...)` before evaluation.

## Implementation provenance

This release reorganizes the supplied MSHNet-based research implementation and retains its attention backbone, multi-scale prediction heads and SLS objective. The released CFRNet configuration corresponds to the supplied complete B2 + SEP-S3-v2 network. Historical experiment selectors, ablation runners and diagnostic collectors are omitted. Naming and repository structure have been simplified; this does not imply that the underlying backbone or inherited loss was newly invented here.
