"""Pluggable LLM judge hook.

Rules catch numbers, dates and explicit negations. Anything subtler
("I live with my parents" vs "my apartment lease renews in June") needs a
real semantic judgment call. Register your own judge function; it runs on
entry pairs the rule engine did *not* flag.

The judge signature::

    def my_judge(text_a: str, text_b: str) -> dict | None:
        # return {"contradiction": True, "explanation": "..."} or None

Example::

    from contradiction_judge import set_judge

    def gpt_judge(a, b):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user",
                       "content": f"Do these two memory facts contradict each "
                                  f"other? Answer JSON {{contradiction, explanation}}.\n\n1. {a}\n2. {b}"}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    set_judge(gpt_judge)

CLI: ``contradiction-judge check memory.json --hook mymodule:my_judge``.
"""

from __future__ import annotations

import importlib
from typing import Callable

_judge: Callable | None = None


def set_judge(fn: Callable) -> None:
    """Register an LLM judge function. Pass None to disable."""
    global _judge
    _judge = fn


def get_judge() -> Callable | None:
    """Return the currently registered judge, or None."""
    return _judge


def load_hook(spec: str) -> Callable:
    """Load a judge from a ``module:function`` spec, e.g. ``mymod:my_judge``."""
    if ":" not in spec:
        raise ValueError(
            f"hook spec must look like 'module:function', got {spec!r}"
        )
    module_name, func_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise ValueError(f"{spec!r} is not callable")
    return fn
