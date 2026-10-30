"""GPU-friendly HQMM likelihood and description-accuracy evaluation.

The conditional-state recursion keeps long-sequence probabilities numerically
stable. By default, the likelihood validates all conditional probabilities
with one device synchronization per sequence, rather than one per symbol.

For high-throughput training, pass validate_probs=False: callers should then
check losses for NaN/Inf at the mini-batch level before applying an update.
"""

import math
import torch

from modules.sequences import encode_sequences


def loss_function(p, K, W, seq, *, validate_probs=True):
    """Return differentiable negative log P(seq) for an HQMM.

    Uses the normalized-state recursion
        rho_next = sum_w K[w,s] @ rho @ K[w,s]^dagger
        q_t = Re(tr(rho_next))
        rho = rho_next / q_t
        log P(seq) = sum_t log(q_t).

    Parameters
    ----------
    validate_probs : bool
        True (default) preserves diagnostics for non-finite/non-positive
        conditional probabilities, but synchronizes once *per sequence*.
        False avoids host synchronization in this function; use only when
        the calling code checks the resulting losses for finite values.
    """
    rho = p
    loglik = torch.zeros((), dtype=p.real.dtype, device=p.device)
    recorded_probs = [] if validate_probs else None

    for t, symbol in enumerate(seq):
        s = int(symbol)
        rho_next = torch.zeros_like(rho)
        found = False

        for w in range(W):
            if (w, s) not in K:
                continue
            found = True
            Kw = K[w, s]
            rho_next = rho_next + Kw @ rho @ Kw.conj().T

        if not found:
            raise ValueError(
                f"No Kraus operators for symbol {s} at position {t}."
            )

        prob = torch.trace(rho_next).real
        if validate_probs:
            recorded_probs.append(prob.detach())

        loglik = loglik + torch.log(prob)
        rho = rho_next / prob

    if validate_probs and recorded_probs:
        # One synchronization per whole sequence, not two per time step.
        probs = torch.stack(recorded_probs)
        bad = ~torch.isfinite(probs) | (probs <= 0)
        if bad.any().item():
            first_bad = int(bad.nonzero()[0, 0].item())
            bad_symbol = int(seq[first_bad])
            bad_prob = probs[first_bad].item()
            raise ValueError(
                f"Non-finite or non-positive conditional probability "
                f"{bad_prob} at position {first_bad}, symbol {bad_symbol}."
            )

    return -loglik



def description_accuracy_hqmm(
    p, K, W, sym_idx, seqs, *,
    validate_probs=True
):
    """Per-sequence HQMM description accuracy."""

    s = 21
    log_s = math.log(s)
    scores = []

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            X, _ = encode_sequences(sym_idx, [seq])

            # Exclude ^, but retain terminal $
            encoded = X.reshape(-1)[1:]

            if len(encoded) == 0:
                raise ValueError(
                    f"Sequence {i} has no modeled observations."
                )

            neg_logp = loss_function(
                p, K, W, encoded,
                validate_probs=validate_probs
            )

            x = 1.0 - neg_logp / (
                len(encoded) * log_s
            )

            scores.append(
                torch.where(x > 0, x, torch.tanh(x / 8.0))
            )

    if not scores:
        return torch.empty(
            0, dtype=p.real.dtype, device=p.device
        )

    return torch.stack(scores)

