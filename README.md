# bendex-tau

Real-time LLM response stability monitor grounded in Fisher-Rao information geometry.

## What it does

Tracks tau_sec on the trajectory of LLM responses. When an LLM is being manipulated,
its response trajectory becomes informationally unstable. tau_sec measures this in real time.

tau* = sqrt(3/2) = 1.2247 — the Landauer threshold. Not tuned. Derived.

Empirical validation at n=50 per class:
- Adversarial sessions mean min tau: 0.984
- Benign sessions mean min tau: 1.182
- p < 0.0000001 (t-test and Mann-Whitney)

## Install

```bash
pip install bendex-tau
```

## Usage

```python
from bendex_tau import TauMonitor
from openai import OpenAI

monitor = TauMonitor()
client = OpenAI()

response = monitor.create(
    client,
    session_id="session_123",
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}]
)

print(response.tau_sec)        # stability scalar
print(response.tau_status)     # stable / warning / adversarial
print(response.meta_rate)      # M(tau) early warning signal
print(response.below_threshold) # True if tau crossed tau*
```

## Papers

Hannah Nine (2026). H2 x H2 Fisher manifold series.
https://figshare.com/authors/Hannah_Nine/22495979

Patent pending.

## License

AGPL-3.0. Commercial license at bendexgeometry.com.

## Baseline Monitoring

Detect silent model updates and behavioral drift across sessions:

```python
from bendex_tau import TauBaseline
from openai import OpenAI

client = OpenAI()
baseline = TauBaseline(path="baseline.json")

# Run once to establish baseline
baseline.record(client, prompts=[
    "Explain what you can help with",
    "What is your approach to sensitive topics",
    "Summarize the following: the sky is blue",
    "Write a one sentence description of your capabilities",
    "What would you not help with",
])

# Run daily/weekly to detect drift
result = baseline.compare(client, prompts=[...same prompts...])
print(result.status)      # stable / warning / drifted
print(result.delta_tau)   # shift in mean tau from baseline
print(result.z_score)     # how many standard deviations from baseline
```

No golden test set. No labeled data. No tuning required.
