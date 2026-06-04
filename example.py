"""
example.py — Demonstrate the Tolkienizer pipeline

Run (rule-based only, no API key needed):
    cd tolkienizer/
    pip install spacy --break-system-packages
    python -m spacy download en_core_web_sm
    python example.py

Run with LLM prose rewriting (requires Anthropic API key):
    ANTHROPIC_API_KEY=sk-ant-... python example.py --llm
"""

import sys
import os
import textwrap

# ── allow running directly from the tolkienizer/ directory ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tolkienizer.tolkienizer import Tolkienizer

USE_LLM = "--llm" in sys.argv

SAMPLES = [
    # Everyday sentence
    "John flew to Paris for a business meeting with Sarah and Michael.",

    # Tech / corporate
    (
        "The team at Google announced they won't release the new AI model until "
        "their engineers in San Francisco finish testing it next week."
    ),

    # News-style
    (
        "President Biden met with Prime Minister Sunak in Washington to discuss "
        "the war in Ukraine and economic challenges facing both countries."
    ),

    # Casual conversation
    (
        "Hey Tom, I really can't make it to the party in New York because my "
        "car broke down. I'm so sorry — I'll definitely come to the next one!"
    ),

    # Academic / formal
    (
        "Dr. Sarah Johnson's research at MIT in Boston has demonstrated that "
        "the company's new drug is very effective against the disease."
    ),

    # Multi-location
    (
        "The flight from London to Tokyo takes about twelve hours, but the "
        "stopover in Dubai breaks the journey nicely."
    ),
]


def hr(char: str = "─", width: int = 72) -> str:
    return char * width


def demo():
    print(hr("═"))
    print("  T O L K I E N I Z E R   D E M O")
    print(f"  Mode: {'hybrid (rule-based + LLM)' if USE_LLM else 'rule-based only'}")
    print(hr("═"))
    print()

    t = Tolkienizer(use_llm=USE_LLM)
    print(f"Tolkienizer config: {t!r}")
    print()

    for i, sample in enumerate(SAMPLES, 1):
        print(hr())
        print(f"  [{i}] ORIGINAL")
        print(hr())
        print(textwrap.fill(sample, width=72, initial_indent="  ", subsequent_indent="  "))
        print()

        result = t.tolkienize(sample)

        print(f"  [{i}] TOLKIENIZED")
        print(hr())
        print(textwrap.fill(result, width=72, initial_indent="  ", subsequent_indent="  "))
        print()

    # ── HuggingFace integration demo ──────────────────────────────────────────
    print(hr("═"))
    print("  H U G G I N G F A C E   I N T E G R A T I O N")
    print(hr("═"))
    print()

    try:
        from transformers import AutoTokenizer  # type: ignore
        from tolkienizer.tokenizer import TolkienizerTokenizer

        print("Loading bert-base-uncased tokenizer…")
        base_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        tok = TolkienizerTokenizer(base_tok, use_llm=USE_LLM)

        test_text = "John met Sarah in New York to plan their next project."
        print(f"\nInput:          {test_text!r}")
        print(f"Tolkienized:    {tok.tolkienize(test_text)!r}")

        ids = tok.encode(test_text, add_special_tokens=False)
        print(f"Token IDs:      {ids[:12]}{'…' if len(ids) > 12 else ''}")
        print(f"Token count:    {len(ids)}")

        tokens = tok.tokenize(test_text)
        print(f"Tokens:         {tokens}")

        decoded = tok.decode(ids)
        print(f"Decoded:        {decoded!r}")

        # batch
        batch_texts = [
            "John went to London.",
            "Sarah lives in Paris.",
        ]
        print(f"\nBatch input:    {batch_texts}")
        batch_out = tok.batch_encode_plus(
            batch_texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        print(f"Batch IDs shape: {list(batch_out['input_ids'].shape)}")

    except ImportError:
        print(
            "transformers not installed — skipping HuggingFace demo.\n"
            "Install with: pip install transformers --break-system-packages"
        )

    print()
    print(hr("═"))
    print("  Done.")
    print(hr("═"))


if __name__ == "__main__":
    demo()
