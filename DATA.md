# Where the shipped data comes from

Everything written for this project is CC BY 4.0 (see `LICENSE`). The files below came from somewhere else
and keep their own terms.

| File | Source | Licence |
|---|---|---|
| `script/script-a-567.*` | sentences from [`meldynamics/lt-stressed-corpus`](https://huggingface.co/datasets/meldynamics/lt-stressed-corpus) (MATAS + ALKSNIS, Vytautas Magnus University), with the corpus's own normative stress marks | CC BY 4.0 |
| `script/script-b-289.*` | 249 sentences from the same corpus + 40 written for this project to cover brand and product names | CC BY 4.0 |
| `text/lexicon/corpus_lexicon.json` | word forms with exactly one attested stress, extracted from the same corpus (~130k forms) | CC BY 4.0 |
| `text/lexicon/corpus_ambiguous.json` | homograph forms with their attested stress variants, same corpus | CC BY 4.0 |
| `text/lexicon/corpus_bare.json` | forms the corpus leaves unmarked | CC BY 4.0 |
| `text/wikt_lexicon.json` | stress patterns harvested from the Lithuanian Wiktionary | **CC BY-SA 4.0** — Wiktionary's licence, share-alike; if you redistribute a modified copy of this file it stays CC BY-SA |
| `text/stress_overrides.json`, `text/foreign_bare.txt` | written for this project | CC BY 4.0 |

## Deliberately absent

- **`text/names/names_first_vlkk.json`** — first names with stress and declension from the State Commission
  of the Lithuanian Language (vardai.vlkk.lt). Authoritative, and not ours to publish. `kirtis_v2.py` works
  without it: first names fall through to the model, which is right when the stress follows the spelling and
  wrong when it does not. The expected format is `{"Jonas": {"m": "Jõnas"}, ...}`.
- **Surname data** from the Institute of the Lithuanian Language, for the same reason.
- **Our own recordings.** One speaker's read takes are personal data, not a corpus.

## Models this code downloads

| Model | What for | Licence |
|---|---|---|
| [`kristijonas/krapas-Regina`](https://huggingface.co/kristijonas/krapas-Regina) | the voice | CC BY-NC 4.0 (inherited from OmniVoice weights) |
| [`kristijonas/banano-kirtis-v2`](https://huggingface.co/kristijonas/banano-kirtis-v2) | stress marking | CC BY 4.0 |
| [`kristijonas/kmynas-parakeet-lt-v4`](https://huggingface.co/kristijonas/kmynas-parakeet-lt-v4) | checking the takes (default) | CC BY 4.0 |
| [`kristijonas/paprika-whisper-lt-v3`](https://huggingface.co/kristijonas/paprika-whisper-lt-v3) | checking the takes (whisper alternative) | CC BY 4.0 |
| [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice) | the base model you fine-tune | code Apache 2.0, weights CC BY-NC |

Because the OmniVoice weights are CC BY-NC, any voice fine-tuned from them is non-commercial too. A
commercial Lithuanian voice needs a base with an Apache or MIT licence, trained from scratch — a different
project.
