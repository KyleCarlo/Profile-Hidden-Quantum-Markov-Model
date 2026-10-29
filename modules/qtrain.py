"""Mini-batch COSM HQMM training with CUDA-friendly likelihood evaluation.

Install alongside qevaluation_gpu.py (renamed to modules/qevaluation.py).
"""

import time

import numpy as np
import pandas as pd
import torch

from modules.qevaluation import description_accuracy_hqmm, loss_function
from modules.sequences import encode_sequences


def stack_matrices(A):
    keys = sorted(A.keys(), key=lambda k: (k[1], k[0]))
    return torch.cat([A[key] for key in keys], dim=0)


def construct_UV(G_stacked, kappa):
    U = torch.cat([G_stacked, kappa], dim=1)
    V = torch.cat([kappa, -G_stacked], dim=1)
    return U, V


def kappa_tau(tau, kappa, U, V):
    I = torch.eye(U.shape[1], dtype=U.dtype, device=U.device)
    A = I + (tau / 2) * (V.conj().T @ U)
    B = V.conj().T @ kappa
    return kappa - tau * U @ torch.linalg.solve(A, B)


def get_dict_from_stacked(stacked_K, keys):
    keys = sorted(keys, key=lambda k: (k[1], k[0]))
    n = stacked_K.shape[1]
    return {
        key: stacked_K[i * n:(i + 1) * n, :]
        for i, key in enumerate(keys)
    }


def train_hqmm(
    p, K, W, sym_idx, train_seqs,
    val_seqs=None, test_seqs=None,
    max_iter=200, tol=1e-3,
    tau=0.001, alpha=0.92,
    batch_size=8, verbose=True,
    device=None,
):
    """Train HQMM with mini-batch COSM and the Wen-Yin retraction.

    train_time is wall-clock time of training batches (excluding evaluations).
    CUDA is selected automatically when available, unless device is given.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if not K:
        raise ValueError("K must not be empty")

    device = torch.device(
        device if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    keys = sorted(K.keys(), key=lambda k: (k[1], k[0]))
    kraus_dtype = K[keys[0]].dtype
    if any(K[key].dtype != kraus_dtype for key in keys):
        raise ValueError("All Kraus operators must have the same dtype")

    p = p.detach().to(device=device, dtype=kraus_dtype)
    K = {
        key: K[key].detach().clone().to(device=device)
        for key in keys
    }

    if verbose:
        print(f"Training on {device} (dtype={kraus_dtype})", flush=True)

    # Symbol indices may stay on CPU; GPU work occurs on p and K.
    train_encoded = []
    for seq in train_seqs:
        X, _ = encode_sequences(sym_idx, [seq])
        encoded = X.reshape(-1)[1:]
        if len(encoded) > 0:
            train_encoded.append(encoded)
    if not train_encoded:
        raise ValueError("No nonempty training sequences")

    n_train = len(train_encoded)
    n_batches = (n_train + batch_size - 1) // batch_size
    use_val = val_seqs is not None and len(val_seqs) > 0
    history = []
    train_time = 0.0
    current_tau = tau

    def params():
        return {key: value.detach().clone() for key, value in K.items()}

    def record(it):
        with torch.no_grad():
            ll_total = torch.zeros((), dtype=p.real.dtype, device=device)
            for seq in train_encoded:
                # The epoch-level log-likelihood is checked once at the end.
                ll_total -= loss_function(
                    p, K, W, seq, validate_probs=False
                )
            ll = ll_total.item()
            if not np.isfinite(ll):
                raise ValueError(f"Non-finite training log-likelihood at epoch {it}")

            row = {
                "iter": it,
                "train_time": train_time,
                "loglik": ll,
            }

            for name, seqs in (("val", val_seqs), ("test", test_seqs)):
                if seqs is not None and len(seqs) > 0:
                    da = description_accuracy_hqmm(
                        p, K, W, sym_idx, seqs, validate_probs=False
                    )
                    # DA is already a real tensor on the correct device.
                    # A single host transfer retrieves both statistics.
                    mean_da, std_da = torch.stack((
                        da.mean(), da.std(correction=0)
                    )).tolist()
                    if not (np.isfinite(mean_da) and np.isfinite(std_da)):
                        raise ValueError(
                            f"Non-finite {name} description accuracy at epoch {it}"
                        )
                    row[f"{name}_da_mean"] = mean_da
                    row[f"{name}_da_std"] = std_da

        history.append(row)
        if verbose:
            extra = "".join(
                f"  {key} {value:.4f}"
                for key, value in row.items()
                if key.endswith("da_mean")
            )
            print(
                f"iter {it:3d}  "
                f"train {train_time:7.1f}s  "
                f"log-lik {ll:.2f}{extra}",
                flush=True,
            )
        return row

    initial = record(0)
    best = (initial["val_da_mean"], params()) if use_val else None

    for it in range(1, max_iter + 1):
        indices = np.random.permutation(n_train)
        for batch_idx, start in enumerate(range(0, n_train, batch_size), start=1):
            batch_indices = indices[start:start + batch_size]
            batch_count = len(batch_indices)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            K = {
                key: K[key].detach().clone().requires_grad_(True)
                for key in keys
            }

            detached_losses = []
            for idx in batch_indices:
                seq = train_encoded[idx]
                L = loss_function(p, K, W, seq, validate_probs=False)
                detached_losses.append(L.detach())
                (L / batch_count).backward()

            # One GPU-to-CPU synchronization for the entire batch.
            loss_values = torch.stack(detached_losses).cpu().numpy()
            bad_positions = np.flatnonzero(~np.isfinite(loss_values))
            if bad_positions.size:
                bad_idx = int(batch_indices[bad_positions[0]])
                raise ValueError(
                    f"Non-finite loss at iteration {it}, "
                    f"batch {batch_idx}, sequence {bad_idx}"
                )
            batch_loss = float(np.mean(loss_values, dtype=np.float64))

            G = {
                key: (
                    K[key].grad / 2
                    if K[key].grad is not None
                    else torch.zeros_like(K[key])
                )
                for key in keys
            }

            kappa = stack_matrices(K).detach()
            G_stacked = stack_matrices(G).detach()
            U, V = construct_UV(G_stacked, kappa)

            with torch.no_grad():
                new_kappa = kappa_tau(current_tau, kappa, U, V)
                K = get_dict_from_stacked(new_kappa, keys)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_time += time.perf_counter() - t0

            if verbose:
                print(
                    f"  iter {it:3d}  "
                    f"batch {batch_idx:3d}/{n_batches}  "
                    f"loss {batch_loss:.4f}  "
                    f"tau {current_tau:.6f}",
                    flush=True,
                )

        row = record(it)

        if use_val and row["val_da_mean"] > best[0]:
            best = (row["val_da_mean"], params())

        previous_ll = history[-2]["loglik"]
        current_ll = row["loglik"]
        if (
            np.isfinite(previous_ll)
            and np.isfinite(current_ll)
            and abs(current_ll - previous_ll) < tol
        ):
            break

        current_tau *= alpha

    if use_val:
        K = best[1]

    return K, pd.DataFrame(history)
