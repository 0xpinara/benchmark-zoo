"""One training run, and the per-instance record it leaves behind.

The unit of output is a boolean vector over test nodes: was this node
classified correctly.  That vector, not the scalar accuracy, is what the
instance-level bootstrap of :mod:`bzoo.resample.instance` needs, and storing it
means the whole downstream analysis can be redone without retraining.

Model selection is on validation accuracy, with the test set touched exactly
once per run, at the epoch chosen by validation.  That is the discipline the
benchmark asks for, and it also means the numbers we produce are the same kind
of numbers a leaderboard entry reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .loaders import NodeDataset, normalise_adjacency
from .models import build_model, n_parameters, propagation_mode, sparse_to_torch


@dataclass
class RunResult:
    arch: str
    dataset: str
    config: Dict[str, object]
    seed: int
    epochs: int
    best_epoch: int
    eval_every: int
    valid_accuracy: float
    test_accuracy: float
    train_accuracy: float
    n_parameters: int
    seconds: float
    correct: np.ndarray = field(repr=False)  # bool over test nodes

    def to_record(self) -> Dict[str, object]:
        from .tuning import config_id

        return {
            "arch": self.arch,
            "dataset": self.dataset,
            "config_id": config_id(self.config),
            "seed": self.seed,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "eval_every": self.eval_every,
            "valid_accuracy": self.valid_accuracy,
            "test_accuracy": self.test_accuracy,
            "train_accuracy": self.train_accuracy,
            "n_parameters": self.n_parameters,
            "seconds": self.seconds,
            **{f"cfg_{k}": v for k, v in self.config.items()},
        }


_ADJ_CACHE: Dict[str, torch.Tensor] = {}


def _get_adj(ds: NodeDataset, mode: str, device: torch.device) -> torch.Tensor:
    key = f"{ds.name}:{mode}:{device}"
    if key not in _ADJ_CACHE:
        _ADJ_CACHE[key] = sparse_to_torch(
            normalise_adjacency(ds.adj, mode=mode), device
        )
    return _ADJ_CACHE[key]


def pick_device(prefer: str = "auto") -> torch.device:
    """CUDA if present, otherwise CPU.

    Apple's MPS backend is skipped on purpose: as of PyTorch 2.8 it has no
    sparse kernels, so ``torch.sparse.mm`` falls over, and a dense adjacency is
    not an option for the larger graph.  The runs here are full-batch on graphs
    of at most 170,000 nodes, which CPU handles in seconds to a minute.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_once(
    ds: NodeDataset,
    arch: str,
    config: Dict[str, object],
    seed: int,
    epochs: int = 300,
    patience: int = 100,
    eval_every: int = 1,
    device: Optional[torch.device] = None,
) -> RunResult:
    """Train one model and return its per-node test correctness.

    ``epochs`` is fixed across architectures, so the training budget is equal
    by construction.  ``patience`` stops a run early only when validation
    accuracy has not improved for that many epochs; because the budget is
    counted in epochs and not in wall clock, early stopping does not give any
    architecture an advantage.

    ``eval_every`` evaluates on a stride rather than every epoch.  Evaluation
    is a second full forward pass, so on the large graph it is half the cost of
    the run.  The stride is the same for every architecture and is recorded in
    the result, so it cannot advantage one of them; it does mean the selected
    epoch is on a grid, which costs a little accuracy on all of them equally.
    """
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    x = torch.from_numpy(ds.features).to(device)
    y = torch.from_numpy(ds.labels).to(device)
    adj = _get_adj(ds, propagation_mode(arch), device)
    tr = torch.from_numpy(ds.train_idx).to(device)
    va = torch.from_numpy(ds.valid_idx).to(device)
    te = torch.from_numpy(ds.test_idx).to(device)

    model = build_model(arch, ds.n_features, ds.n_classes, config).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_val = -1.0
    best_epoch = -1
    best_correct = None
    best_train = 0.0
    since_best = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(x, adj)
        loss = F.cross_entropy(out[tr], y[tr])
        loss.backward()
        opt.step()

        if epoch % eval_every != 0 and epoch != epochs:
            continue
        model.eval()
        with torch.no_grad():
            logits = model(x, adj)
            pred = logits.argmax(dim=1)
            val_acc = float((pred[va] == y[va]).float().mean().item())
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_correct = (pred[te] == y[te]).detach().cpu().numpy()
                best_train = float((pred[tr] == y[tr]).float().mean().item())
                since_best = 0
            else:
                since_best += eval_every
                if since_best >= patience:
                    break

    assert best_correct is not None
    return RunResult(
        arch=arch,
        dataset=ds.name,
        config=dict(config),
        seed=seed,
        epochs=epochs,
        best_epoch=best_epoch,
        eval_every=eval_every,
        valid_accuracy=best_val,
        test_accuracy=float(best_correct.mean()),
        train_accuracy=best_train,
        n_parameters=n_parameters(model),
        seconds=round(time.time() - t0, 2),
        correct=best_correct.astype(bool),
    )
