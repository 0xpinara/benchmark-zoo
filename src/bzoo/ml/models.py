"""Node classification models, written out in plain PyTorch.

Three families and their ablations.  All of them are full-batch: the graphs
here fit in memory, and full-batch training removes minibatch sampling as a
source of variance we would then have to model.

``mlp``    per-node classifier, no graph at all
``gcn``    Kipf and Welling (2017): ``H' = act(A_hat H W)``
``sage``   Hamilton, Ying and Leskovec (2017), mean aggregator:
           ``H' = act(H W_self + A_mean H W_neigh)``

The ablations remove one component at a time, which is what makes them a
plausible null population for "a paper that reports an architectural
improvement":

``gcn`` with ``propagation="none"``
    the graph is replaced by the identity, so the model becomes an MLP with
    the same parameter count and training budget
``sage`` with ``use_neighbour=False``
    the aggregation branch is removed, keeping the self branch
``*`` with ``n_layers=1``
    depth removed
``gcn`` with ``propagation="unnormalised"``
    the degree normalisation removed, keeping the propagation

Every ablation keeps the parameter count and the training budget of the model
it ablates, so the comparison is not confounded by capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_to_torch(a: sp.csr_matrix, device: torch.device) -> torch.Tensor:
    coo = a.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, coo.shape, device=device).coalesce()


class MLP(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        n_layers: int = 2,
        dropout: float = 0.5,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = [n_features] + [n_hidden] * (n_layers - 1) + [n_classes]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.norms = nn.ModuleList(
            (nn.BatchNorm1d(dims[i + 1]) if batch_norm else nn.Identity())
            for i in range(n_layers - 1)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x


class GCN(nn.Module):
    """Graph convolution.  ``propagate=False`` is the no-graph ablation."""

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        n_layers: int = 2,
        dropout: float = 0.5,
        batch_norm: bool = False,
        propagate: bool = True,
    ) -> None:
        super().__init__()
        dims = [n_features] + [n_hidden] * (n_layers - 1) + [n_classes]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.norms = nn.ModuleList(
            (nn.BatchNorm1d(dims[i + 1]) if batch_norm else nn.Identity())
            for i in range(n_layers - 1)
        )
        self.dropout = dropout
        self.propagate = propagate

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        for i, layer in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
            if self.propagate:
                x = torch.sparse.mm(adj, x)
            if i < len(self.layers) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x


class SAGE(nn.Module):
    """Mean-aggregator GraphSAGE.  ``use_neighbour=False`` is the ablation."""

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        n_layers: int = 2,
        dropout: float = 0.5,
        batch_norm: bool = False,
        use_neighbour: bool = True,
    ) -> None:
        super().__init__()
        dims = [n_features] + [n_hidden] * (n_layers - 1) + [n_classes]
        self.self_lin = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.neigh_lin = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.norms = nn.ModuleList(
            (nn.BatchNorm1d(dims[i + 1]) if batch_norm else nn.Identity())
            for i in range(n_layers - 1)
        )
        self.dropout = dropout
        self.use_neighbour = use_neighbour

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        for i in range(len(self.self_lin)):
            x = F.dropout(x, p=self.dropout, training=self.training)
            h = self.self_lin[i](x)
            if self.use_neighbour:
                h = h + self.neigh_lin[i](torch.sparse.mm(adj, x))
            x = h
            if i < len(self.self_lin) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x


# Registry: name -> (class, propagation mode, extra kwargs, is_ablation, ablates)
ARCHITECTURES: Dict[str, dict] = {
    "mlp": {"cls": MLP, "prop": "none", "kwargs": {}, "ablation_of": None},
    "gcn": {"cls": GCN, "prop": "sym", "kwargs": {}, "ablation_of": None},
    "sage": {"cls": SAGE, "prop": "row", "kwargs": {}, "ablation_of": None},
    "gcn_noprop": {
        "cls": GCN,
        "prop": "sym",
        "kwargs": {"propagate": False},
        "ablation_of": "gcn",
    },
    "gcn_unnorm": {
        "cls": GCN,
        "prop": "none",
        "kwargs": {},
        "ablation_of": "gcn",
    },
    "sage_noneigh": {
        "cls": SAGE,
        "prop": "row",
        "kwargs": {"use_neighbour": False},
        "ablation_of": "sage",
    },
}


def build_model(
    arch: str,
    n_features: int,
    n_classes: int,
    config: Dict[str, object],
) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}")
    spec = ARCHITECTURES[arch]
    kwargs = dict(spec["kwargs"])
    return spec["cls"](
        n_features=n_features,
        n_hidden=int(config["n_hidden"]),
        n_classes=n_classes,
        n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]),
        batch_norm=bool(config["batch_norm"]),
        **kwargs,
    )


def propagation_mode(arch: str) -> str:
    return ARCHITECTURES[arch]["prop"]


def n_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
