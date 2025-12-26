# NNUE feature export and training notes

This engine now exposes a minimal NNUE-style feature extractor and reference network so you can prototype training in Python and later re-use weights in C++.

## Feature extraction

- Function: `skaks_eval.features_from_fen(fen: str) -> numpy.ndarray[int8]`
- Shape: `(1537,)` with values in {0, 1}
  - Layout: 12 piece kinds × 64 squares × 2 king buckets = 1536 bits, plus one side-to-move bit at the end.
  - King buckets are split: white-king-relative bucket first, black-king-relative bucket second.
- Example:

```python
import numpy as np
from skaks_eval import features_from_fen

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
feat = features_from_fen(fen)
assert feat.shape == (1537,)
```

## Reference network shape

The C++ helper `chess::NnueNetwork` expects:

- `w1`: flattened row-major `[hidden, input]` (float)
- `b1`: `[hidden]`
- `w2`: `[hidden]`
- `b2`: scalar

The forward pass applies a ReLU hidden layer and a single linear output. Input size is fixed to 1537; hidden size is inferred from `b1.size()`.

## Quick PyTorch training sketch

This is a bare-bones example using FEN/outcome pairs (outcome in [0, 1], e.g., win=1, loss=0, draw=0.5). Adjust to your dataset and regularization needs.

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skaks_eval import features_from_fen

class FenDataset(Dataset):
    def __init__(self, rows):  # rows: list of (fen, outcome)
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        fen, outcome = self.rows[idx]
        x = torch.from_numpy(features_from_fen(fen)).float()
        y = torch.tensor(outcome, dtype=torch.float32)
        return x, y

class TinyNnue(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(1537, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)

def train(rows, hidden=128, epochs=3, batch_size=1024, lr=1e-3):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ds = FenDataset(rows)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = TinyNnue(hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid inside

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    return model
```

## Exporting weights to C++ layout

After training, move tensors to CPU and flatten:

```python
state = model.state_dict()
w1 = state['fc1.weight'].contiguous().view(-1).tolist()
b1 = state['fc1.bias'].tolist()
w2 = state['fc2.weight'].contiguous().view(-1).tolist()  # shape [1, hidden]
b2 = float(state['fc2.bias'][0])
```

These arrays map directly onto `chess::NnueNetwork` fields. The current C++ side only runs inference; weight loading/writing is left to the caller (e.g., embed JSON/binary or extend bindings to accept weights).

## Dataset tips

- If using engine-eval-derived outcomes, keep them in [0, 1] and consider label smoothing to reduce overconfidence.
- De-duplicate positions and enforce minimum ply to avoid trivial book openings (use `tuning/filter_texel_dataset.py`).
- Shuffle data and monitor loss on a held-out split; prefer large batch sizes for stable gradients.

## Next steps

- Add a Python binding to accept serialized weights and call the C++ `NnueNetwork` forward for evaluation parity.
- Wire a loader in the engine that reads on-disk weights and swaps them into the search stack.
