#!/usr/bin/env python3
"""Kirčiuoklis v2 layer 1: RULES + OVERRIDES on top of the neural model.

Why: the neural model generalises by word shape and context, but two classes
are structurally beyond it — foreign-origin proper names (no Lithuanian family
to generalise from; it correctly reports LOW confidence, e.g. Jakubsonas 0.698)
and productive suffix families where it is inconsistent (Kazláuskas correct,
Petrauskás wrong — same suffix). Both are exactly what VDU's rule system
handles well, so we borrow the mechanism rather than the data.

Resolution order (first hit wins):
  1. overrides.json   user-curated, authoritative (KJ can add any name)
  2. lexicon          161k attested forms from svogunas/g2p-lt-lexicon
  3. suffix rules     productive families with fixed accentual status
  4. neural model     everything else / context disambiguation

Layer 4 also keeps the last word when the model is CONFIDENT and the rule is
generic — rules override only where they are known-reliable (1-3).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OVERRIDES = HERE / "overrides.json"
LEXICON = HERE / "lexicon_labels.json"     # compact {word: [syl, acc]}

# Productive suffix families. syl_from_end: which syllable counting BACK from
# the word end carries the stress (1 = last, 2 = penultimate, 3 = ante-).
# Sources: LT grammar suffix accentuation (VDU T-41/05 report families).
SUFFIX_RULES = [
    ("evičius", 3, "tilde"),    # Vaitkẽvičius, Adamkẽvičius
    ("avičius", 3, "tilde"),
    ("auskas",  2, "acute"),    # Kazláuskas, Petráuskas
    ("auskaitė", 3, "acute"),
    ("auskienė", 3, "acute"),
    ("aitis",   2, "acute"),    # Jonáitis
    ("aitė",    2, "acute"),
    ("ienė",    2, "acute"),    # Petráuskienė
    ("iūtė",    2, "acute"),
    ("ūnas",    2, "acute"),    # Baltrū́nas
    ("ynas",    2, "acute"),
    ("iškis",   2, "short"),    # Vilkavìškis
    ("iškių",   2, "short"),
    ("ukas",    2, "short"),
    ("ukė",     2, "short"),
    ("ingas",   2, "short"),
    ("inga",    2, "short"),
    ("ierius",  2, "tilde"),
    ("ierė",    2, "tilde"),
    # Latinate -orius borrowings: stress on the syllable BEFORE -ius, short.
    # Systematic error found on the hard set 2026-08-07 (we said tilde on the
    # wrong syllable for procesorius/profesorius; VDU: procèsorius, profèsorius).
    # from_end 3, NOT 2: nuclei are o-e-o-iu, so 2 lands on the wrong o
    # and produced procesòrius while VDU (and this comment) say
    # procèsorius. Wrong since the rule was added 2026-08-07; the
    # -torius family below genuinely is 2 (direktòrius).
    ("sorius",  3, "short"),    # procèsorius, profèsorius, akcèsorius
    ("torius",  2, "short"),    # direktòrius, inspektòrius, rektòrius
    ("orius",   2, "short"),
    ("sorė",    2, "short"),
    ("torė",    2, "short"),
    # -ùmas abstract nouns from adjectives: stress on the -ùm- syllable.
    # Found on the 2026-08-09 VDU comparison (we said nuosaĩkumo, VDU
    # nuosaikùmo). Safe against rū́mas/krū́mas/dū́mas — those carry ū, not u.
    ("umas",    2, "short"),    # gerùmas, didùmas, nuosaikùmas
    ("umo",     2, "short"),
    ("umą",     2, "short"),
    ("umu",     2, "short"),
    ("ume",     2, "short"),
    ("umai",    2, "short"),
    ("umų",     2, "short"),
    ("umus",    2, "short"),
]

# Words whose accent depends on MEANING, not spelling. The lexicon holds one
# entry each and outranks the model, so letting it answer makes context
# disambiguation structurally impossible — "jis yrà geras" (būti) and "pūva ir
# ỹra" (irti) would always get the same mark. These defer to the model, the
# only layer that sees the sentence. Membership is EARNED: a word belongs here
# only if the model demonstrably differentiates it.
#
# EMPTY ON PURPOSE. Tested 2026-08-10 with "yra" (yrà from būti vs ỹra from
# irti), the canonical LT case: the model answered yrà at conf 1.0 in ALL
# FOUR contexts, including "medis pūva ir yra" where ỹra is correct. Deferring
# therefore bought nothing and cost the lexicon's authority, so the word went
# back. The mechanism stays because it is the right shape once a model can
# actually do this; the blocker is training data, not layer order — LIEPA-3
# is realized speech and the irti reading is vanishingly rare in it.
HOMOGRAPHS: set[str] = set()

VOW = set("aeiouyąęėįųū")
SON = set("jlmnr")
# Mixed diphthongs (al, en, ir, um…) form on SHORT vowels only; after a long
# vowel the sonorant is a plain coda — dė̃l, not dėl̃.
SHORT_VOW = set("aeiu")
# The ONLY true diphthongs. Any other vowel pair is hiatus = two syllables:
# te-a-tras, i-de-a-las, Iz-ra-e-lis. Merging those misplaced the accent on
# every Latinate loan (teatrãs for teãtras) until 2026-08-09.
DIPH = {"ai", "au", "ei", "ui", "ie", "uo", "oi", "ou", "eu"}
# vowels that a preceding consonant+i palatalises into one syllable (ia io iu iū iuo)
GLIDE_VOW = set("aąoųuū")


def nucleus_groups(word: str) -> list[list[int]]:
    """Syllable nuclei as character-index groups. SINGLE SOURCE OF TRUTH —
    the renderer and the suffix-rule indexer must agree, so both import this
    (two hand-kept copies is how the accent placement drifted before)."""
    w = word.lower()
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, c in enumerate(w):
        nxt = w[i + 1] if i + 1 < len(w) else ""
        if c in VOW:
            if not cur:
                cur = [i]
                continue
            prev = w[cur[-1]]
            # `i` before a vowel is a palatalization glide (ia, io, iu, iau)
            # when a consonant precedes it — one syllable, not two.
            # Only before a/ą/o/u/ų/ū: i+i, i+ė, i+e are hiatus (pri-i-ma,
            # pasi-ė-mė) — merging them mislabelled 300+ corpus forms (2026-09-06).
            glide = (len(cur) == 1 and prev == "i" and cur[0] > 0
                     and w[cur[0] - 1] not in VOW and c in GLIDE_VOW)
            if prev + c in DIPH or glide:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        elif cur and c in SON and nxt not in VOW and w[cur[-1]] in SHORT_VOW:
            cur.append(i)
            groups.append(cur)
            cur = []
        else:
            if cur:
                groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def nucleus_count(word: str) -> int:
    """Number of syllable nuclei (by definition, the renderer's convention)."""
    return len(nucleus_groups(word))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class RuleLayer:
    def __init__(self):
        self.overrides = _load(OVERRIDES)
        self.lexicon = _load(LEXICON)

    def lookup(self, word: str) -> tuple[int, str, str] | None:
        """(syllable_index, accent, source) or None to defer to the model."""
        wl = word.lower()
        if wl in self.overrides:
            o = self.overrides[wl]
            return o["syllable"], o["accent"], "override"
        if wl in HOMOGRAPHS:
            return None          # only the model sees the sentence
        if wl in self.lexicon:
            syl, acc = self.lexicon[wl]
            return syl, acc, "lexicon"
        for suf, from_end, acc in SUFFIX_RULES:
            if wl.endswith(suf) and len(wl) > len(suf):
                n = nucleus_count(wl)
                idx = n - from_end
                if 0 <= idx < n:
                    return idx, acc, f"suffix:-{suf}"
        return None

    def add_override(self, word: str, syllable: int, accent: str) -> None:
        self.overrides[word.lower()] = {"syllable": syllable, "accent": accent}
        OVERRIDES.write_text(json.dumps(self.overrides, ensure_ascii=False,
                                        indent=1), encoding="utf-8")
