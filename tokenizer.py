"""
tokenizer.py — HuggingFace tokenizer wrapper with Tolkienizer pre-processing

Wraps any HuggingFace PreTrainedTokenizer (or PreTrainedTokenizerFast) so that
the Tolkienizer transform is applied transparently before tokenisation.

Usage
-----
    from transformers import AutoTokenizer
    from tolkienizer.tolkienizer import Tolkienizer
    from tokenizer import TolkienizerTokenizer

    base = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok  = TolkienizerTokenizer(base, use_llm=False)

    # Standard HF interface — tolkienization is invisible to the caller
    ids    = tok.encode("John went to London.")
    tokens = tok.tokenize("John went to London.")
    batch  = tok.batch_encode_plus(["John went to London.", "Sarah lives in Paris."])

    # You can also inspect the tolkienized intermediate text
    middle = tok.tolkienize("John went to London.")
    print(middle)  # → "Aragorn went to Minas Morgul."

Architecture
------------
TolkienizerTokenizer does NOT subclass PreTrainedTokenizer because HF's
metaclass machinery makes that fragile across tokenizer variants.  Instead it
uses __getattr__ delegation so any attribute not explicitly defined here falls
through to the underlying tokenizer, making it a transparent drop-in.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from tolkienizer.tolkienizer import Tolkienizer


class TolkienizerTokenizer:
    """
    A drop-in wrapper for any HuggingFace tokenizer that applies
    the Tolkienizer transform before tokenisation.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
        Any HuggingFace tokenizer instance.
    tolkienizer : Tolkienizer, optional
        Pre-configured Tolkienizer instance.  If not supplied one is
        created from the remaining kwargs.
    use_llm : bool
        Passed to Tolkienizer if *tolkienizer* is None.
    api_key : str, optional
        Anthropic API key passed to Tolkienizer.
    llm_model : str
        Claude model string passed to Tolkienizer.
    spacy_model : str
        spaCy model string passed to Tolkienizer.
    """

    def __init__(
        self,
        tokenizer,
        tolkienizer: Optional[Tolkienizer] = None,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        llm_model: str = "claude-haiku-4-5-20251001",
        spacy_model: str = "en_core_web_sm",
    ):
        # Store with mangled names so __getattr__ doesn't intercept them
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(
            self,
            "_tolkienizer",
            tolkienizer
            or Tolkienizer(
                use_llm=use_llm,
                api_key=api_key,
                llm_model=llm_model,
                spacy_model=spacy_model,
            ),
        )

    # ── Delegation ─────────────────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying HF tokenizer."""
        return getattr(object.__getattribute__(self, "_tokenizer"), name)

    def __repr__(self) -> str:
        tok = object.__getattribute__(self, "_tokenizer")
        tlk = object.__getattribute__(self, "_tolkienizer")
        return f"TolkienizerTokenizer(tokenizer={tok!r}, tolkienizer={tlk!r})"

    # ── Tolkienize helpers ─────────────────────────────────────────────────────

    def tolkienize(self, text: str) -> str:
        """Return the tolkienized intermediate text without tokenising."""
        tlk = object.__getattribute__(self, "_tolkienizer")
        return tlk.tolkienize(text)

    def _t(self, text: Union[str, list]) -> Union[str, list]:
        """Apply tolkienize to str or list[str]."""
        tlk = object.__getattribute__(self, "_tolkienizer")
        if isinstance(text, str):
            return tlk.tolkienize(text)
        return [tlk.tolkienize(t) for t in text]

    # ── Core tokenizer interface ───────────────────────────────────────────────

    def __call__(
        self,
        text: Union[str, list[str], None] = None,
        text_pair: Union[str, list[str], None] = None,
        **kwargs,
    ) -> Any:
        """
        Tolkienize then tokenise.  Mirrors the HF tokenizer __call__ signature.
        """
        tok = object.__getattribute__(self, "_tokenizer")
        t_text      = self._t(text)      if text      is not None else None
        t_text_pair = self._t(text_pair) if text_pair is not None else None
        return tok(t_text, t_text_pair, **kwargs)

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        **kwargs,
    ) -> list[int]:
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.encode(self._t(text), text_pair, **kwargs)

    def encode_plus(
        self,
        text: str,
        text_pair: Optional[str] = None,
        **kwargs,
    ) -> dict:
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.encode_plus(self._t(text), text_pair, **kwargs)

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            list[str],
            list[tuple[str, str]],
        ],
        **kwargs,
    ) -> dict:
        tok = object.__getattribute__(self, "_tokenizer")
        tlk = object.__getattribute__(self, "_tolkienizer")

        # Handle plain list[str] and list[(str, str)] pairs
        processed: list = []
        for item in batch_text_or_text_pairs:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                processed.append((tlk.tolkienize(item[0]), tlk.tolkienize(item[1])))
            else:
                processed.append(tlk.tolkienize(item))

        return tok.batch_encode_plus(processed, **kwargs)

    def tokenize(self, text: str, **kwargs) -> list[str]:
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.tokenize(self._t(text), **kwargs)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Decode passes straight through — no inverse tolkienization."""
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.decode(token_ids, **kwargs)

    def batch_decode(self, sequences: list[list[int]], **kwargs) -> list[str]:
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.batch_decode(sequences, **kwargs)

    def convert_tokens_to_ids(self, tokens: Union[str, list[str]]) -> Any:
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(
        self,
        ids: Union[int, list[int]],
        skip_special_tokens: bool = False,
    ) -> Any:
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, save_directory: str, **kwargs) -> Any:
        """Save the underlying tokenizer (tolkienizer state is not persisted)."""
        tok = object.__getattribute__(self, "_tokenizer")
        return tok.save_pretrained(save_directory, **kwargs)

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def base_tokenizer(self):
        """Direct access to the wrapped HF tokenizer."""
        return object.__getattribute__(self, "_tokenizer")

    @property
    def tolkienizer_instance(self) -> Tolkienizer:
        """Direct access to the Tolkienizer transform."""
        return object.__getattribute__(self, "_tolkienizer")
