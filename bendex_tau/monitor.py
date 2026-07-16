from .session import TauSession
from typing import Dict, Optional

class TauResponse:
    def __init__(self, openai_response, tau_data: dict):
        self._response      = openai_response
        self.tau_sec        = tau_data.get("tau_sec")
        self.tau_star       = tau_data.get("tau_star")
        self.tau_status     = tau_data.get("tau_status")
        self.meta_rate      = tau_data.get("meta_rate")
        self.below_threshold = tau_data.get("below_threshold", False)
        self.turns          = tau_data.get("turns")
        self.session_id     = tau_data.get("session_id")
        self._tau_data      = tau_data

    def __getattr__(self, name):
        return getattr(self._response, name)

    def __repr__(self):
        return (
            f"TauResponse(tau_sec={self.tau_sec}, "
            f"tau_status={self.tau_status!r}, "
            f"meta_rate={self.meta_rate}, "
            f"turns={self.turns})"
        )


class TauMonitor:
    """
    Wraps an OpenAI-compatible client to track tau_sec on response trajectories.

    Usage:
        from bendex_tau import TauMonitor
        from openai import OpenAI

        monitor = TauMonitor()
        client = OpenAI()

        response = monitor.create(
            client,
            session_id="user_123",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}]
        )

        print(response.tau_sec)
        print(response.tau_status)
        print(response.meta_rate)
    """

    def __init__(self):
        self._sessions: Dict[str, TauSession] = {}

    def _get_session(self, session_id: str) -> TauSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = TauSession(session_id)
        return self._sessions[session_id]

    def create(self, client, session_id: str = "default", **kwargs) -> TauResponse:
        response = client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        tau_data = self._get_session(session_id).update(text)
        return TauResponse(response, tau_data)

    def get_session(self, session_id: str) -> Optional[TauSession]:
        return self._sessions.get(session_id)

    def reset_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]
