"""contradiction-judge: semantic contradiction detection for AI memory systems.

The memory auditor: given a list of memory entries (facts the system has
recorded about the user), find pairs that contradict each other and suggest
how to arbitrate the conflict.

Combines:
- rule-based detectors (numeric / date / boolean / preference conflicts)
- a pluggable LLM judge hook for semantic contradictions rules cannot catch
"""

__version__ = "0.1.0"

from .detect import check_memories
from .hooks import set_judge, get_judge, load_hook

__all__ = ["__version__", "check_memories", "set_judge", "get_judge", "load_hook"]
