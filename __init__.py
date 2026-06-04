"""
tolkienizer — a Tolkien-register pre-processing modifier for tokenizers.

Quickstart
----------
    from tolkienizer import Tolkienizer
    from tolkienizer.tokenizer import TolkienizerTokenizer

    # Rule-based only (no API key needed)
    t = Tolkienizer(use_llm=False)
    print(t("John flew to Paris for a meeting."))

    # Hybrid: rule-based + LLM prose rewriting
    t = Tolkienizer(use_llm=True, api_key="sk-ant-...")
    print(t("John flew to Paris for a meeting."))

    # Wrapped around a HuggingFace tokenizer
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok  = TolkienizerTokenizer(base, use_llm=False)
    ids  = tok.encode("John went to London.")
"""

from tolkienizer.tolkienizer import Tolkienizer
from tolkienizer.tokenizer import TolkienizerTokenizer

__all__ = ["Tolkienizer", "TolkienizerTokenizer"]
__version__ = "0.1.0"
