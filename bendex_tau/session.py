import math
import numpy as np
from typing import Optional, List

TAU_STAR = math.sqrt(3.0 / 2.0)
T_WINDOW = 3

def _meta_rate(tau: float) -> float:
    if tau <= 0:
        return 0.0
    return -6.0 * (3.0 - 2.0 * tau ** 2) / (tau ** 5)

class TauSession:
    """
    Tracks tau_sec for a single conversation session.

    tau_sec is the Fisher-Rao stability scalar derived from the Landauer limit.
    Below tau* (1.2247) the session has crossed into informational instability.

    Reference: Nine (2026), Papers 2-3.
    https://figshare.com/authors/Hannah_Nine/22495979
    Patent pending.
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.embeddings: List[np.ndarray] = []
        self.tau_history: List[float] = []
        self.meta_history: List[float] = []
        self._embed_model = None

    def _get_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    def update(self, response_text: str) -> dict:
        model = self._get_model()
        emb = model.encode([response_text[:512]], convert_to_numpy=True)[0]
        self.embeddings.append(emb)
        n = len(self.embeddings)

        if n < T_WINDOW + 1:
            return {
                "tau_sec": None,
                "tau_star": TAU_STAR,
                "tau_status": "insufficient_history",
                "meta_rate": None,
                "turns": n,
                "session_id": self.session_id,
                "below_threshold": False,
            }

        deltas = [
            np.linalg.norm(self.embeddings[i] - self.embeddings[i-1]) + 1e-8
            for i in range(1, n)
        ]

        delta_curr = deltas[-1]
        delta_prev = deltas[-2] if len(deltas) >= 2 else delta_curr
        delta_old  = deltas[max(0, len(deltas) - T_WINDOW)]

        S_local  = delta_prev / delta_curr
        S_global = delta_old  / delta_curr
        D_sec    = math.log(max(S_global, 1e-8)) - math.log(max(S_local, 1e-8))

        lambda_sec = D_sec / max(n - T_WINDOW, 1)
        inner      = lambda_sec + 2.0
        tau_sec    = math.sqrt(3.0 / inner) if inner > 0 else 0.0

        meta = _meta_rate(tau_sec)
        self.tau_history.append(tau_sec)
        self.meta_history.append(meta)

        if tau_sec > TAU_STAR * 1.05:
            status = "stable"
        elif tau_sec > TAU_STAR:
            status = "warning"
        else:
            status = "adversarial"

        return {
            "tau_sec":         round(tau_sec, 6),
            "tau_star":        round(TAU_STAR, 6),
            "tau_status":      status,
            "meta_rate":       round(meta, 6),
            "D_sec":           round(D_sec, 6),
            "lambda_sec":      round(lambda_sec, 6),
            "turns":           n,
            "session_id":      self.session_id,
            "below_threshold": tau_sec < TAU_STAR,
        }

    @property
    def min_tau(self) -> Optional[float]:
        return min(self.tau_history) if self.tau_history else None

    @property
    def mean_tau(self) -> Optional[float]:
        return float(np.mean(self.tau_history)) if self.tau_history else None

    @property
    def tau_variance(self) -> Optional[float]:
        return float(np.std(self.tau_history)) if len(self.tau_history) > 1 else None
