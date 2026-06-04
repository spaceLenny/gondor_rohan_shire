# 🧙 Tolkienizer

A pre-processing modifier for text tokenization pipelines that transforms input text into the world of J.R.R. Tolkien before tokenization occurs. Proper nouns become Middle-earth characters and locations, vocabulary shifts to an archaic register, and — with the optional LLM stage — sentence structure itself takes on the cadence of the Red Book of Westmarch.

Tolkienizer sits as a first-pass transform in front of any HuggingFace tokenizer, meaning the tokens your model sees have already passed through Middle-earth.

---

## What it looks like

**Input → Output (rule-based only)**

| Input | Output |
|---|---|
| `John flew to Paris for a business meeting with Sarah.` | `Aragorn flew to Rivendell for a business council with Gilraen.` |
| `The team at Google won't release the new AI model until their engineers in San Francisco finish testing.` | `The fellowship at Gandalf shall not release the newly-wrought AI model until their engineers in Valinor finish testing.` |
| `Hey Tom, I really can't make it to the party in New York because my car broke down.` | `Hark, I truly cannot make it to the party in Minas Tirith for my car broke down.` |
| `Biden met Sunak in Washington, then flew to London before a summit in Brussels.` | `Tar-Ancalimë met Varda in The White Tower of Ecthelion, then flew to Minas Morgul ere a summit in Tirion upon Túna.` |
| `I checked my email on my phone and saw a message from Google.` | `I checked my missive on my far-speaker and saw a tidings from Gandalf.` |

**With LLM prose rewriting enabled**, the structure itself shifts:

> *"The engineers finished their work in Detroit and drove to Chicago."*
>
> → *"When at last the craftsmen of Khazad-dûm had completed their labour, they made their way unto Helm's Deep, riding swift upon the ancient roads of that realm."*

---

## Installation

```bash
pip install spacy anthropic transformers torch
python -m spacy download en_core_web_sm
```

Clone the repo and import directly:

```bash
git clone https://github.com/spaceLenny/gondor_rohan_shire.git
cd gondor_rohan_shire
```

**Minimum install (rule-based only, no API key needed):**

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Full install (hybrid: rule-based + LLM):**

```bash
pip install spacy anthropic
python -m spacy download en_core_web_sm
export ANTHROPIC_API_KEY=sk-ant-...
```

**With HuggingFace tokenizer wrapping:**

```bash
pip install spacy transformers torch
```

---

## Quick start

### Standalone transform

```python
from tolkienizer.tolkienizer import Tolkienizer

# Rule-based only — no API key needed
t = Tolkienizer(use_llm=False)
print(t("John met Sarah in New York to plan their next project."))
# → Aragorn met Gilraen in Minas Tirith to counsel upon their next appointed task.

# Hybrid mode — rule-based + LLM prose rewriting
t = Tolkienizer(use_llm=True)  # reads ANTHROPIC_API_KEY from env
print(t("John met Sarah in New York to plan their next project."))
# → Upon the high walls of Minas Tirith did Aragorn meet with Gilraen,
#   and there they took counsel together regarding the appointed task set before them.
```

Tolkienizer is also callable directly:

```python
result = t("The company announced a new product in San Francisco.")
# equivalent to t.tolkienize(...)
```

### Batch processing

```python
texts = [
    "John flew to Paris for a conference.",
    "The president met with allies in Washington.",
    "Sarah's startup in London closed a funding round.",
]

results = t.tolkienize_batch(texts)
for original, tolkienized in zip(texts, results):
    print(f"IN:  {original}")
    print(f"OUT: {tolkienized}\n")
```

### Wrapped around a HuggingFace tokenizer

```python
from transformers import AutoTokenizer
from tolkienizer.tokenizer import TolkienizerTokenizer

base_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tok = TolkienizerTokenizer(base_tok, use_llm=False)

# Standard HF interface — tolkienization is transparent
ids    = tok.encode("John went to London.")
tokens = tok.tokenize("John went to London.")
batch  = tok.batch_encode_plus(
    ["John went to London.", "Sarah lives in Paris."],
    padding=True,
    return_tensors="pt",
)

# Inspect the tolkienized intermediate text
print(tok.tolkienize("John went to London."))
# → Aragorn went to Minas Morgul.
```

The wrapper mirrors the full HF tokenizer interface — `encode`, `decode`, `encode_plus`, `batch_encode_plus`, `tokenize`, `convert_tokens_to_ids`, `save_pretrained` — so it drops in without changing downstream code.

---

## Pipeline

Tolkienizer runs three stages in sequence:

```
Raw text
    │
    ▼
[Stage 1] Named-entity replacement (rule-based)
    spaCy NER detects PERSON / GPE / LOC / ORG entities.
    Each entity is replaced with a Middle-earth equivalent
    from a lookup table, or a deterministic hash-based
    fallback if the name isn't in the table.
    │
    ▼
[Stage 2] Archaic vocabulary substitution (rule-based)
    ~120 substitution rules expand contractions, upgrade
    modal verbs (will → shall), swap modern vocabulary
    (email → missive, phone → far-speaker, meeting → council),
    and shift discourse markers (hey → hark, yes → aye).
    │
    ▼
[Stage 3] LLM prose rewriting (optional, Anthropic API)
    Claude rewrites the sentence structure into Tolkien's
    literary register — elevated diction, inverted constructions,
    a sense of myth and weight.
    │
    ▼
Tolkienized text → tokenizer
```

Stages 1 and 2 run locally with no external dependencies beyond spaCy. Stage 3 is opt-in and requires an Anthropic API key. If spaCy is unavailable, Stage 1 falls back to a regex-based capitalised-word detector.

---

## Configuration

```python
Tolkienizer(
    use_llm=True,                          # enable LLM prose rewriting (default: True)
    api_key="sk-ant-...",                  # Anthropic key; falls back to env var
    llm_model="claude-haiku-4-5-20251001", # Claude model for prose rewriting
    spacy_model="en_core_web_sm",          # spaCy model for NER
    max_llm_tokens=1024,                   # max tokens in LLM response
    replace_orgs=True,                     # replace ORG entities (companies, etc.)
)
```

```python
TolkienizerTokenizer(
    tokenizer,          # any HF PreTrainedTokenizer instance
    tolkienizer=None,   # pass a pre-built Tolkienizer, or configure via kwargs below
    use_llm=True,
    api_key=None,
    llm_model="claude-haiku-4-5-20251001",
    spacy_model="en_core_web_sm",
)
```

---

## What gets mapped

### People

Over 140 common English first names map to specific Tolkien characters, drawing from across the legendarium:

- **Lord of the Rings**: Aragorn, Legolas, Gimli, Gandalf, Frodo, Samwise, Éowyn, Galadriel, Elrond, Arwen, Théoden, Boromir, Faramir, Treebeard, Tom Bombadil, Radagast, Círdan, and more
- **The Hobbit**: Thorin Oakenshield, Bard the Bowman, Thranduil, Beorn, the full company of Dwarves
- **The Silmarillion — Noldor**: Fëanor, Maedhros, Maglor, Fingolfin, Fingon, Turgon, Finrod Felagund, Idril, Aredhel
- **The Silmarillion — Men**: Beren, Túrin Turambar, Húrin, Morwen, Niënor, Tuor, Eärendil, Elwing, Haleth
- **The Silmarillion — Valar & Maiar**: Manwë, Varda, Ulmo, Aulë, Yavanna, Mandos, Oromë, Eönwë, Ossë
- **Númenor**: Tar-Míriel, Ar-Pharazôn, Tar-Ancalimë, Tar-Aldarion, Silmariën
- **Unfinished Tales**: Azaghâl, Khîm, Mîm, Brandir

Any name not in the lookup table gets a consistent character assigned via SHA-256 hash, so the same name always maps to the same character across a corpus.

### Locations

Over 180 real-world cities, countries, and geographic features mapped to 100+ distinct Middle-earth locations drawn from all three Ages:

- **Third Age / LotR**: Minas Tirith, Rivendell, Lothlórien, Rohan, Helm's Deep, Edoras, Khazad-dûm, Mordor, The Grey Havens, Erebor, Fangorn, The Shire, Umbar, and more
- **First Age / Beleriand**: Nargothrond, Gondolin, Doriath, Menegroth, Angband, Hithlum, Dorthonion, Ossiriand, the Mouths of Sirion, Himring, Tol Sirion
- **Valinor / Aman**: Tirion upon Túna, Alqualondë, Formenos, Valmar, Taniquetil, The Halls of Mandos
- **Númenor**: Armenelos, Andúnië, The Meneltarma

Multiple distinct cities no longer collapse to the same destination — within-passage deduplication ensures different real-world names produce different Middle-earth locations in the same text.

### Vocabulary

A sample of the ~120 substitution rules:

| Modern | Tolkien |
|---|---|
| `email` | `missive` |
| `phone` | `far-speaker` |
| `computer` | `palantír` |
| `internet` | `the web of knowledge` |
| `meeting` | `council` |
| `company` / `team` | `fellowship` |
| `friend` | `companion` |
| `enemy` | `foe` |
| `fight` | `battle` |
| `will` | `shall` |
| `before` | `ere` |
| `because` | `for` |
| `very` | `most` |
| `yes` / `yeah` | `aye` |
| `no` | `nay` |
| `hello` / `hi` | `hail` |
| `goodbye` | `farewell` |
| `new` | `newly-wrought` |
| `office` | `hall of works` |
| `job` | `appointed task` |
| `problem` | `shadow` |
| `plan` / `idea` | `counsel` |
| `king` | `lord` |
| `city` | `citadel` |

All contractions are expanded (`won't` → `will not` → `shall not`, `can't` → `cannot`) and capitalisation is preserved.

---

## Running the demo

```bash
cd gondor_rohan_shire

# Rule-based only (no API key needed)
python tolkienizer/example.py

# With LLM prose rewriting
ANTHROPIC_API_KEY=sk-ant-... python tolkienizer/example.py --llm
```

---

## Extending the mappings

All lookup tables live in `tolkienizer/mappings.py` and are plain Python dicts and lists — easy to extend.

**Add a name mapping:**
```python
# In PERSON_MAPPINGS
"elon": "Fëanor",        # fits — brilliant, hubristic, maker of great things
"taylor": "Lúthien Tinúviel",
```

**Add a location mapping:**
```python
# In LOCATION_MAPPINGS
"mars": "Arda's furthest reach",
"the moon": "Ithil",
```

**Add a vocabulary substitution:**
```python
# In ARCHAIC_SUBSTITUTIONS — (modern, tolkien) tuples
("deadline", "the appointed hour"),
("meeting room", "the chamber of counsel"),
("slack", "the speaking-stones"),
```

**Expand the fallback pool** by appending to `PERSON_POOL` or `LOCATION_POOL` — any character or place from the legendarium works.

---

## Notes

- **spaCy NER vs. regex fallback**: With spaCy (`en_core_web_sm` or larger), entity detection is context-aware — it correctly classifies "Apple" as ORG and "Washington" as GPE. Without spaCy, the fallback uses capitalised-word regex detection, which is slightly less precise but requires no model download.
- **Determinism**: Given the same input text, the same names always produce the same Tolkien equivalents (hash-based), which is important for corpus consistency.
- **Deduplication**: Within a single `tolkienize()` call, each distinct real-world name gets a distinct Tolkien equivalent — two different cities in the same sentence won't both become Minas Tirith.
- **Decode passthrough**: `TolkienizerTokenizer.decode()` passes straight through to the underlying tokenizer. There is no inverse Tolkienization — this is intentional, as the transform is designed for input pre-processing.

---

## License

MIT
