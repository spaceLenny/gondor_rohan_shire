"""
tolkienizer.py — Core Tolkienizer transform

Pipeline (hybrid approach):
  1. spaCy NER  →  replace PERSON / GPE / LOC / ORG entities with Middle-earth equivalents
  2. Rule-based archaic substitutions  →  expand contractions, swap vocabulary
  3. (optional) LLM prose rewriting  →  elevate sentence structure to Tolkien register

Usage
-----
    from tolkienizer import Tolkienizer

    t = Tolkienizer(use_llm=False)            # rule-based only
    t = Tolkienizer(use_llm=True, api_key="sk-ant-...")  # hybrid

    result = t.tolkienize("John met Sarah in New York to discuss their plan.")
    # → "Aragorn met Galadriel in Minas Tirith to discourse upon their counsel."
"""

from __future__ import annotations

import re
import hashlib
import logging
from typing import Optional

from tolkienizer.mappings import (
    PERSON_POOL,
    PERSON_MAPPINGS,
    LOCATION_POOL,
    LOCATION_MAPPINGS,
    ORG_POOL,
    ARCHAIC_SUBSTITUTIONS,
)

logger = logging.getLogger(__name__)

# ── spaCy lazy-load ────────────────────────────────────────────────────────────
_nlp = None

def _get_nlp(model: str = "en_core_web_sm"):
    """Lazy-load spaCy model; falls back gracefully if unavailable."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy  # type: ignore
        _nlp = spacy.load(model)
    except Exception as exc:
        logger.warning(
            "spaCy model '%s' unavailable (%s). "
            "Falling back to regex-based NER. "
            "Install with: python -m spacy download %s",
            model, exc, model,
        )
        _nlp = None
    return _nlp


# ── Regex fallback NER ─────────────────────────────────────────────────────────
_CAPITALIZED_WORD = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

# Words that are capitalised only because they open a sentence, not because
# they are proper nouns — skip these in the regex-NER fallback.
_COMMON_CAPS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "it", "he", "she", "they", "we", "i",
    "but", "and", "or", "so", "yet", "nor",
    "in", "on", "at", "by", "to", "of", "for", "with",
    "my", "your", "his", "her", "our", "their", "its",
    "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "not", "no", "yes",
})


class _Entity:
    """Minimal stand-in for a spaCy span."""
    __slots__ = ("text", "label_", "start_char", "end_char")

    def __init__(self, text: str, label_: str, start_char: int, end_char: int):
        self.text = text
        self.label_ = label_
        self.start_char = start_char
        self.end_char = end_char


def _regex_ner(text: str) -> list[_Entity]:
    """
    Lightweight fallback: treat every capitalised word/phrase as PERSON
    unless it matches a known location key.
    """
    entities: list[_Entity] = []
    for m in _CAPITALIZED_WORD.finditer(text):
        word = m.group(0)
        low = word.lower()
        # Skip common words that are only capitalised because they open a sentence
        if low in _COMMON_CAPS:
            continue
        if low in LOCATION_MAPPINGS:
            label = "GPE"
        else:
            label = "PERSON"
        entities.append(_Entity(word, label, m.start(), m.end()))
    return entities


# ── Deterministic hash-based pool lookup ───────────────────────────────────────

def _hash_pick(text: str, pool: list[str], used: set[str] | None = None) -> str:
    """
    Pick a pool entry deterministically from text using SHA-256.

    If *used* is provided, skip entries already chosen in this passage
    so that multiple different real-world names don't collapse to the
    same Middle-earth destination/character.
    """
    digest = int(hashlib.sha256(text.lower().encode()).hexdigest(), 16)
    if used is None:
        return pool[digest % len(pool)]
    # Walk forward through the pool (with wrap-around) until we find an unused entry
    for offset in range(len(pool)):
        candidate = pool[(digest + offset) % len(pool)]
        if candidate not in used:
            return candidate
    # All entries used (shouldn't happen in practice) — fall back to hash
    return pool[digest % len(pool)]


# ── Archaic substitution engine ────────────────────────────────────────────────

def _apply_archaic_substitutions(text: str) -> str:
    """
    Apply ARCHAIC_SUBSTITUTIONS as whole-word, case-aware replacements.
    Preserves the original word's capitalisation (Title / ALL CAPS / lower).
    """
    for original, replacement in ARCHAIC_SUBSTITUTIONS:
        # Build pattern: whole-word, case-insensitive
        pattern = r"(?<!\w)" + re.escape(original) + r"(?!\w)"
        flags = re.IGNORECASE

        def _replace(m: re.Match, rep: str = replacement, orig: str = original) -> str:
            matched = m.group(0)
            if matched.isupper():
                return rep.upper()
            if matched[0].isupper() and not matched.isupper():
                return rep.capitalize()
            return rep

        text = re.sub(pattern, _replace, text, flags=flags)
    return text


# ── Main Tolkienizer class ─────────────────────────────────────────────────────

class Tolkienizer:
    """
    Transforms input text into Tolkien-register prose.

    Parameters
    ----------
    use_llm : bool
        Whether to call the Anthropic API for prose rewriting (step 3).
        Set to False for pure rule-based operation with no API dependency.
    api_key : str, optional
        Anthropic API key.  Falls back to the ANTHROPIC_API_KEY env var.
    llm_model : str
        Which Claude model to call.  Defaults to claude-haiku-4-5 for speed.
    spacy_model : str
        spaCy model name.  Defaults to en_core_web_sm.
    max_llm_tokens : int
        Max tokens for the LLM prose rewriting response.
    replace_orgs : bool
        Whether to replace ORG entities with Tolkien fellowship names.
    """

    def __init__(
        self,
        use_llm: bool = True,
        api_key: Optional[str] = None,
        llm_model: str = "claude-haiku-4-5-20251001",
        spacy_model: str = "en_core_web_sm",
        max_llm_tokens: int = 1024,
        replace_orgs: bool = True,
    ):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.max_llm_tokens = max_llm_tokens
        self.spacy_model = spacy_model
        self.replace_orgs = replace_orgs
        self._client = None

        if use_llm:
            self._init_llm_client(api_key)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _init_llm_client(self, api_key: Optional[str]):
        try:
            import anthropic  # type: ignore
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        except ImportError:
            logger.warning(
                "anthropic package not installed. Disabling LLM rewriting. "
                "Install with: pip install anthropic"
            )
            self.use_llm = False

    def _map_person(self, name: str, used: set[str] | None = None) -> str:
        """Map a person name to a Tolkien character."""
        first = name.split()[0].lower()
        if first in PERSON_MAPPINGS:
            result = PERSON_MAPPINGS[first]
        elif name.lower() in PERSON_MAPPINGS:
            result = PERSON_MAPPINGS[name.lower()]
        else:
            result = _hash_pick(name, PERSON_POOL, used)
        # If the direct mapping collides with an already-used character,
        # find the next available one from the pool.
        if used is not None and result in used:
            result = _hash_pick(name, PERSON_POOL, used)
        return result

    def _map_location(self, loc: str, used: set[str] | None = None) -> str:
        """Map a location name to a Middle-earth location."""
        low = loc.lower().strip()
        if low in LOCATION_MAPPINGS:
            result = LOCATION_MAPPINGS[low]
        else:
            # Partial match: check if any key is contained in the location name
            result = next(
                (v for k, v in LOCATION_MAPPINGS.items() if k in low),
                None,
            )
            if result is None:
                result = _hash_pick(loc, LOCATION_POOL, used)
        # Deduplicate within the passage
        if used is not None and result in used:
            result = _hash_pick(loc, LOCATION_POOL, used)
        return result

    def _map_org(self, org: str, used: set[str] | None = None) -> str:
        """Map an organisation name to a Tolkien fellowship/group."""
        return _hash_pick(org, ORG_POOL, used)

    def _extract_and_replace_entities(self, text: str) -> str:
        """
        Step 1: Use spaCy (or regex fallback) to find named entities and
        replace them with Tolkien equivalents.
        """
        nlp = _get_nlp(self.spacy_model)

        if nlp is not None:
            doc = nlp(text)
            entities = list(doc.ents)
        else:
            entities = _regex_ner(text)  # type: ignore[assignment]

        if not entities:
            return text

        # Build replacement map, deduplicating by text so the same name
        # always maps to the same Tolkien name within a passage.
        # Separate used-sets for people and locations so they don't
        # cross-contaminate each other's deduplication.
        replacement_map: dict[str, str] = {}
        used_people: set[str] = set()
        used_locations: set[str] = set()
        used_orgs: set[str] = set()

        for ent in entities:
            raw = ent.text
            if raw in replacement_map:
                continue  # already resolved
            label = ent.label_

            if label == "PERSON":
                mapped = self._map_person(raw, used_people)
                used_people.add(mapped)
                replacement_map[raw] = mapped
            elif label in ("GPE", "LOC", "FAC"):
                mapped = self._map_location(raw, used_locations)
                used_locations.add(mapped)
                replacement_map[raw] = mapped
            elif label == "ORG" and self.replace_orgs:
                mapped = self._map_org(raw, used_orgs)
                used_orgs.add(mapped)
                replacement_map[raw] = mapped
            # NORP (nationalities), DATE, CARDINAL, etc. — leave as-is

        # Apply replacements longest-first to avoid partial matches
        result = text
        for original in sorted(replacement_map, key=len, reverse=True):
            target = replacement_map[original]
            # Whole-word replacement (handles punctuation boundaries)
            result = re.sub(
                r"(?<!\w)" + re.escape(original) + r"(?!\w)",
                target,
                result,
            )

        return result

    def _rewrite_prose(self, text: str) -> str:
        """
        Step 3: Call Claude to elevate the prose into Tolkien's register.
        Returns the original text unchanged if the API call fails.
        """
        if not self.use_llm or self._client is None:
            return text

        system_prompt = (
            "You are a master of J.R.R. Tolkien's literary style. "
            "Your task is to rewrite the given text so it reads as if it were "
            "written by Tolkien himself — elevated diction, archaic phrasing, "
            "inverted sentence structures where fitting, a sense of myth and weight, "
            "and the cadence of the Red Book of Westmarch. "
            "Preserve the meaning and all proper nouns exactly as given. "
            "Do NOT add new characters or locations. "
            "Return only the rewritten text, no commentary."
        )

        try:
            response = self._client.messages.create(
                model=self.llm_model,
                max_tokens=self.max_llm_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            logger.warning("LLM prose rewriting failed: %s. Returning rule-based output.", exc)
            return text

    # ── Public API ─────────────────────────────────────────────────────────────

    def tolkienize(self, text: str) -> str:
        """
        Transform *text* through the full Tolkienizer pipeline:
          1. Named-entity replacement  (rule-based)
          2. Archaic vocabulary substitution  (rule-based)
          3. Prose register elevation  (LLM, if enabled)

        Parameters
        ----------
        text : str
            Raw input text (any subject matter).

        Returns
        -------
        str
            The Tolkienized text.
        """
        if not text or not text.strip():
            return text

        # Step 1 — entity replacement
        text = self._extract_and_replace_entities(text)

        # Step 2 — archaic substitutions
        text = _apply_archaic_substitutions(text)

        # Step 3 — LLM prose rewriting
        if self.use_llm:
            text = self._rewrite_prose(text)

        return text

    def tolkienize_batch(self, texts: list[str]) -> list[str]:
        """
        Tolkienize a list of texts.  LLM calls are made individually
        (batch endpoint not used to keep latency predictable).

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        list[str]
        """
        return [self.tolkienize(t) for t in texts]

    def __call__(self, text: str) -> str:
        """Alias for tolkienize — allows use as a callable transform."""
        return self.tolkienize(text)

    def __repr__(self) -> str:
        return (
            f"Tolkienizer(use_llm={self.use_llm}, "
            f"model={self.llm_model!r}, "
            f"spacy={self.spacy_model!r})"
        )
