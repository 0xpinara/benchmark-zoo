"""Node classification datasets, loaded as plain numpy and scipy.

Four datasets, one task (transductive node classification), one metric
(accuracy).  Keeping the task and metric fixed is what makes the four
comparable, and the reason they are here is that they sit at very different
distances from their own ceiling, which is what the saturation analysis needs.

``cora``, ``citeseer``, ``pubmed``
    The Planetoid citation networks, with the fixed public split of Yang,
    Cohen and Salakhutdinov (2016): 20 labelled nodes per class for training,
    500 for validation, 1,000 for test.  Loaded from the original release
    files rather than through a graph library, so there is no dependency that
    could silently change the split.
``ogbn-arxiv``
    Open Graph Benchmark (Hu et al., NeurIPS 2020), with the official
    time-based split.  Loaded through the ``ogb`` package, which is also what
    supplies the leaderboard evaluator.

We deliberately do not use ``torch_geometric``.  The models in
:mod:`bzoo.ml.models` are a few lines of sparse matrix multiplication each,
and writing them out means the training loop is inspectable and the
comparison between a model and its own ablation is exact.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp

from ..paths import CACHE, RAW, ensure_dirs

PLANETOID_URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"
PLANETOID_NAMES = ("cora", "citeseer", "pubmed")
DATASETS = PLANETOID_NAMES + ("ogbn-arxiv",)


@dataclass
class NodeDataset:
    name: str
    features: np.ndarray  # (n_nodes, n_features) float32
    labels: np.ndarray  # (n_nodes,) int64
    adj: sp.csr_matrix  # (n_nodes, n_nodes), symmetric, no self loops
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    n_classes: int

    @property
    def n_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.adj.nnz // 2)

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "n_train": int(self.train_idx.size),
            "n_valid": int(self.valid_idx.size),
            "n_test": int(self.test_idx.size),
        }


def _download_planetoid(name: str) -> None:
    import requests

    ensure_dirs()
    out = RAW / "planetoid"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"):
        fname = f"ind.{name}.{suffix}"
        path = out / fname
        if path.exists():
            continue
        resp = requests.get(f"{PLANETOID_URL}/{fname}", timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)


def _load_pickle(path):
    with open(path, "rb") as fh:
        if sys.version_info >= (3, 0):
            return pickle.load(fh, encoding="latin1")
        return pickle.load(fh)  # pragma: no cover


def load_planetoid(name: str) -> NodeDataset:
    """Load one Planetoid dataset with the fixed public split.

    The release stores the test nodes out of order in ``ind.NAME.test.index``,
    and for CiteSeer some test indices are missing entirely; both are handled
    the way the original code does, by placing rows at their recorded index and
    leaving the gaps as zero feature vectors.  Getting this wrong shifts the
    labels of a few hundred nodes and changes accuracy by a percent or two,
    which is the same order as the effects we are trying to measure.
    """
    if name not in PLANETOID_NAMES:
        raise ValueError(f"name must be one of {PLANETOID_NAMES}")
    _download_planetoid(name)
    base = RAW / "planetoid"

    x = _load_pickle(base / f"ind.{name}.x")
    y = _load_pickle(base / f"ind.{name}.y")
    tx = _load_pickle(base / f"ind.{name}.tx")
    ty = _load_pickle(base / f"ind.{name}.ty")
    allx = _load_pickle(base / f"ind.{name}.allx")
    ally = _load_pickle(base / f"ind.{name}.ally")
    graph = _load_pickle(base / f"ind.{name}.graph")
    test_idx_reorder = np.loadtxt(base / f"ind.{name}.test.index", dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)

    if name == "citeseer":
        # Some isolated test nodes are absent from tx; insert zero rows so that
        # the index space stays contiguous.
        full = np.arange(test_idx_range.min(), test_idx_range.max() + 1)
        missing = np.setdiff1d(full, test_idx_reorder)
        tx_ext = sp.lil_matrix((len(full), x.shape[1]))
        tx_ext[test_idx_range - test_idx_range.min(), :] = tx
        tx = tx_ext.tocsr()
        ty_ext = np.zeros((len(full), y.shape[1]))
        ty_ext[test_idx_range - test_idx_range.min(), :] = ty
        ty = ty_ext
        del missing

    features = sp.vstack([allx, tx]).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = np.asarray(features.todense(), dtype=np.float32)

    labels_onehot = np.vstack([ally, ty])
    labels_onehot[test_idx_reorder, :] = labels_onehot[test_idx_range, :]
    labels = labels_onehot.argmax(axis=1).astype(np.int64)

    n = features.shape[0]
    rows, cols = [], []
    for src, dsts in graph.items():
        for dst in dsts:
            if src < n and dst < n:
                rows.append(src)
                cols.append(dst)
    adj = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    adj = adj.maximum(adj.T)  # symmetrise
    adj.setdiag(0)
    adj.eliminate_zeros()

    train_idx = np.arange(y.shape[0], dtype=np.int64)
    valid_idx = np.arange(y.shape[0], y.shape[0] + 500, dtype=np.int64)
    test_idx = np.sort(test_idx_reorder)

    return NodeDataset(
        name=name,
        features=features,
        labels=labels,
        adj=adj.tocsr(),
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        n_classes=int(labels_onehot.shape[1]),
    )


def _with_legacy_torch_load(fn):
    """Run ``fn`` with ``torch.load`` defaulting to ``weights_only=False``.

    ``ogb`` 1.3.6 caches its processed graph with ``torch.save`` of a plain
    Python dict.  PyTorch 2.6 changed the ``torch.load`` default to
    ``weights_only=True``, which refuses to unpickle that, so loading a cached
    OGB dataset raises.  We relax the flag for exactly the duration of the OGB
    call, and only for a file this process wrote itself into ``data/raw/ogb``.
    """
    import torch

    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    try:
        return fn()
    finally:
        torch.load = original


def load_ogbn_arxiv() -> NodeDataset:
    from ogb.nodeproppred import NodePropPredDataset

    ensure_dirs()
    ds = _with_legacy_torch_load(
        lambda: NodePropPredDataset(name="ogbn-arxiv", root=str(RAW / "ogb"))
    )
    split = ds.get_idx_split()
    graph, labels = ds[0]
    n = int(graph["num_nodes"])
    ei = graph["edge_index"]
    adj = sp.coo_matrix(
        (np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(n, n)
    )
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return NodeDataset(
        name="ogbn-arxiv",
        features=np.asarray(graph["node_feat"], dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64).ravel(),
        adj=adj.tocsr(),
        train_idx=np.asarray(split["train"], dtype=np.int64),
        valid_idx=np.asarray(split["valid"], dtype=np.int64),
        test_idx=np.asarray(split["test"], dtype=np.int64),
        n_classes=int(np.max(labels) + 1),
    )


_CACHE: Dict[str, NodeDataset] = {}


def load_dataset(name: str) -> NodeDataset:
    """Load any of the four datasets, memoised within a process."""
    if name in _CACHE:
        return _CACHE[name]
    if name in PLANETOID_NAMES:
        ds = load_planetoid(name)
    elif name == "ogbn-arxiv":
        ds = load_ogbn_arxiv()
    else:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    _CACHE[name] = ds
    return ds


def normalise_adjacency(
    adj: sp.csr_matrix, mode: str = "sym", add_self_loops: bool = True
) -> sp.csr_matrix:
    """Row- or symmetrically normalised adjacency.

    ``"sym"`` gives ``D^-1/2 (A + I) D^-1/2``, the propagation matrix of Kipf
    and Welling (2017).  ``"row"`` gives ``D^-1 (A + I)``, the neighbourhood
    mean used by GraphSAGE.  ``"none"`` returns ``A + I`` unnormalised, which
    is one of the ablations.
    """
    a = adj.astype(np.float32)
    if add_self_loops:
        a = a + sp.eye(a.shape[0], dtype=np.float32, format="csr")
    if mode == "none":
        return a.tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    if mode == "row":
        d = sp.diags(1.0 / deg)
        return (d @ a).tocsr()
    if mode == "sym":
        d = sp.diags(1.0 / np.sqrt(deg))
        return (d @ a @ d).tocsr()
    raise ValueError("mode must be 'sym', 'row' or 'none'")
