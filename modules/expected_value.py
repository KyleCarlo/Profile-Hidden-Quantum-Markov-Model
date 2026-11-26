import numpy as np
from scipy.stats import gumbel_r

def generate_random_sequences(null_probs, n=200, length=100, seed=42):
    rng = np.random.default_rng(seed)

    amino_acids = list(null_probs.keys())

    probabilities = np.array(
        list(null_probs.values()),
        dtype=float
    )

    # Normalize probabilities
    probabilities /= probabilities.sum()

    return [
        "".join(
            rng.choice(amino_acids, size=length, p=probabilities)
        )
        for _ in range(n)
    ]

def calibrate_forward(random_scores):
    """
    Estimate the 4% upper-tail threshold from random
    Forward log-odds scores measured in bits.
    """
    mu, scale = gumbel_r.fit(random_scores)
    t = gumbel_r.ppf(0.96, loc=mu, scale=scale)
    return t

def forward_evalue(score, t, database_size, h=None, N=None):
    """
    Compute approximate Forward E-value using Eddy (2008).
    score and t must be log-odds bit scores.
    """
    lam = np.log(2.0)

    if h is not None and N is not None:
        lam += 1.44 / (h * N)

    return 0.04 * database_size * np.exp(
        -lam * (score - t)
    )
