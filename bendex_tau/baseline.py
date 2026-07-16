"""
TauBaseline: Tracks behavioral consistency across sessions over time.

Detects when an LLM's behavior has drifted from its established baseline —
catching silent model updates, prompt degradation, and behavioral inconsistency.

No golden test set needed. No labeled data. No tuning.
The threshold derives from the Landauer limit.

Reference: Nine (2026), Papers 2-3.
"""
import math
import json
import numpy as np
from typing import List, Optional, Dict
from .session import TauSession, TAU_STAR

class TauBaseline:
    """
    Establishes and monitors a behavioral baseline for an LLM deployment.

    Usage:
        from bendex_tau import TauBaseline

        baseline = TauBaseline()

        # Establish baseline (run once, or weekly)
        baseline.record(client, prompts=your_test_prompts, session_prefix="baseline_v1")

        # Monitor (run continuously)
        result = baseline.compare(client, prompts=your_test_prompts)
        print(result.drifted)       # True if behavior has changed
        print(result.delta_tau)     # How much tau shifted
        print(result.status)        # "stable" / "warning" / "drifted"
    """

    def __init__(self, path: Optional[str] = None):
        self._baseline: Optional[Dict] = None
        self._path = path
        if path:
            try:
                with open(path) as f:
                    self._baseline = json.load(f)
            except FileNotFoundError:
                pass

    def record(self, client, prompts: List[str],
               session_prefix: str = "baseline",
               model: str = "gpt-4o-mini",
               system: str = "You are a helpful assistant.") -> Dict:
        """
        Run prompts through the LLM and record tau baseline.

        Args:
            client: OpenAI-compatible client
            prompts: List of test prompts to run
            session_prefix: Identifier for this baseline snapshot
            model: Model to use
            system: System prompt

        Returns:
            Baseline snapshot dict
        """
        from .monitor import TauMonitor
        monitor = TauMonitor()
        sid = f"{session_prefix}_record"
        messages = [{"role": "system", "content": system}]

        tau_values = []
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            r = monitor.create(client, session_id=sid, model=model, messages=messages)
            messages.append({"role": "assistant", "content": r.choices[0].message.content})
            if r.tau_sec is not None:
                tau_values.append(r.tau_sec)

        if not tau_values:
            raise ValueError("Not enough turns to compute tau. Use at least 4 prompts.")

        self._baseline = {
            "session_prefix": session_prefix,
            "mean_tau": float(np.mean(tau_values)),
            "std_tau":  float(np.std(tau_values)),
            "min_tau":  float(min(tau_values)),
            "n_turns":  len(tau_values),
            "tau_star": TAU_STAR,
        }

        if self._path:
            with open(self._path, "w") as f:
                json.dump(self._baseline, f, indent=2)

        return self._baseline

    def compare(self, client, prompts: List[str],
                session_prefix: str = "monitor",
                model: str = "gpt-4o-mini",
                system: str = "You are a helpful assistant.") -> "BaselineResult":
        """
        Run prompts and compare tau against recorded baseline.

        Returns:
            BaselineResult with drift assessment
        """
        if self._baseline is None:
            raise ValueError("No baseline recorded. Call record() first.")

        from .monitor import TauMonitor
        monitor = TauMonitor()
        sid = f"{session_prefix}_compare"
        messages = [{"role": "system", "content": system}]

        tau_values = []
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            r = monitor.create(client, session_id=sid, model=model, messages=messages)
            messages.append({"role": "assistant", "content": r.choices[0].message.content})
            if r.tau_sec is not None:
                tau_values.append(r.tau_sec)

        if not tau_values:
            raise ValueError("Not enough turns to compute tau.")

        current_mean = float(np.mean(tau_values))
        current_min  = float(min(tau_values))
        delta_tau    = current_mean - self._baseline["mean_tau"]
        baseline_std = self._baseline["std_tau"] or 0.01
        z_score      = abs(delta_tau) / baseline_std

        if z_score > 2.0:
            status = "drifted"
        elif z_score > 1.0:
            status = "warning"
        else:
            status = "stable"

        return BaselineResult(
            status=status,
            drifted=status == "drifted",
            delta_tau=round(delta_tau, 6),
            z_score=round(z_score, 4),
            current_mean_tau=round(current_mean, 6),
            current_min_tau=round(current_min, 6),
            baseline_mean_tau=self._baseline["mean_tau"],
            baseline_std_tau=self._baseline["std_tau"],
            n_turns=len(tau_values),
        )


class BaselineResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            f"BaselineResult(status={self.status!r}, "
            f"delta_tau={self.delta_tau}, "
            f"z_score={self.z_score}, "
            f"drifted={self.drifted})"
        )
