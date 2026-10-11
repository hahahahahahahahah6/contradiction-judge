# contradiction-judge

The memory auditor: semantic contradiction detection for AI memory systems.

AI agents accumulate facts about the user over time — "you are 20", "you
love pizza", "you live in Pasadena". When a new memory contradicts an old
one ("you are 21", "you hate pizza"), something is wrong: a stale entry, a
misheard sentence, a bug upstream. `contradiction-judge` finds those pairs
and suggests how to arbitrate them.

Two detection layers:

1. **Rule-based** (stdlib only): numeric conflicts ("20 years old" vs "21
   years old", unit-aware: "240 months old" == "20 years old"), date
   conflicts ("birthday May 5, 2006" vs "2006-06-01"), factual conflicts
   (name, location), polarity conflicts ("I love pizza" vs "I hate pizza",
   "I am a student" vs "I am not a student").
2. **Pluggable LLM judge hook**: register your own judge function
   (`set_judge` / `--hook module:func`) for semantic contradictions rules
   cannot see ("I live with my parents" vs "my apartment lease renews in
   June"). The hook only runs on pairs the rules did not flag.

Every reported contradiction carries an **arbitration suggestion**:
recency wins when both entries are timestamped; otherwise it is flagged
for manual review.

## Install

```bash
pip install contradiction-judge
```

Requires Python 3.9+. No dependencies — stdlib only.

## Usage

Feed it a JSON file of memory entries:

```json
[
  {"id": "m1", "text": "I am 20 years old.", "created_at": "2025-03-01T10:00:00"},
  {"id": "m2", "text": "I am 21 years old.", "created_at": "2026-09-15T10:00:00"}
]
```

```bash
contradiction-judge check memory.json
```

Output is a JSON report on stdout:

```json
{
  "entries": 2,
  "pairs_checked": 1,
  "skipped": [],
  "contradictions": [
    {
      "entry_a": "m1",
      "entry_b": "m2",
      "kind": "numeric",
      "engine": "rules",
      "facts": [...],
      "explanation": "'I am 20 years old.' vs 'I am 21 years old.': person.age cannot be both 20.0 and 21.0.",
      "arbitration": {
        "strategy": "recency",
        "recommendation": "keep m2, drop or correct m1",
        "reason": "Both entries are timestamped; the newer record (m2) is more likely to be current. Verify before deleting."
      }
    }
  ]
}
```

Flags:

- `--fail` — exit 1 when any contradiction is found (CI-friendly).
- `--hook MODULE:FUNC` — attach an LLM judge for semantic pairs.

Python API:

```python
from contradiction_judge import check_memories, set_judge

report = check_memories([
    {"id": "m1", "text": "I am 20 years old."},
    {"id": "m2", "text": "I am 21 years old."},
])

def gpt_judge(text_a, text_b):
    # call your model here; return {"contradiction": bool, "explanation": str}
    ...

set_judge(gpt_judge)
report = check_memories(entries)  # semantic pairs also get judged
```

## How it differs from memgovern

[memgovern](https://github.com/hahahahahahahahah6/memgovern) is a
**governance framework for AI agent memory**: it governs where memories come
from — provenance tracking, source trust levels, write policies, retention
rules. It answers *"should this memory be trusted and kept?"*

`contradiction-judge` does one thing memgovern does not: **semantic
fact-contradiction detection**. Given a set of memory entries — trusted or
not — it audits whether they agree with each other, pinpoints the exact
conflicting pair ("you are 20" vs "you are 21"), and suggests how to
arbitrate it. It answers *"do these memories contradict each other, and
which one should win?"*

Think of memgovern as the memory system's **governor** (who may write, what
may stay) and contradiction-judge as its **auditor** (do the records agree).
They compose well: govern with memgovern, audit with contradiction-judge.

## Roadmap

- Embedding-based topic alignment for preference matching ("pizza" vs
  "pizzas", "the Bay Area" vs "San Francisco")
- Confidence scores per contradiction
- Auto-fix mode: rewrite the stale entry instead of just flagging it
- Streaming/watch mode for live memory stores

## License

MIT. See [LICENSE](LICENSE).
