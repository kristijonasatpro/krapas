#!/usr/bin/env python3
"""Kirčiuoklis v2 — corpus lexicon in front of a corpus-trained stress model.

Resolution order per word (first hit wins):
  0. pass-through: single letters, short ALL-CAPS tokens (acronyms), tokens with digits, forms the corpus leaves bare
  1. corpus lexicon   meldynamics/lt-stressed-corpus, forms with ONE attested stress (130k forms)
  2. corpus ambiguous forms (homographs): the model picks among the attested variants in context
     (majority variant if it predicts something unattested)
  3. model for everything else (OOV, names)   — model-v2 = xlm-roberta-large trained on the corpus'
     in-context labels (v2/train_v2.py); the old overrides / svogunas lexicon / suffix rules are
     off (USE_RULES) because they lost to the model on the same set.

Measured 2026-09-06 on 3,000 held-out corpus sentences (28,883 words), string-exact vs VDU
normative stressing (kirciuoklis/eval_ltsc.py): banano v12 stack 85.4 % -> v12+lexicon 97.5 %
-> this 99.4 % (sentence-exact 33 % -> 81 % -> 95 %). Numbers are agreement with the corpus,
not human-verified accuracy.

usage: python kirtis_v2.py "Namas buvo labai aukštas."      (or import stress_text / stress_words)
"""
import json, os, re, sys, unicodedata
from pathlib import Path
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent))
from rules import GLIDE_VOW, SON, RuleLayer, nucleus_groups

HERE = Path(__file__).parent
# the model is downloaded from the Hub on first use; point KIRTIS_MODEL at a local directory to use your own
MODEL_DIR = os.environ.get("KIRTIS_MODEL", "kristijonas/banano-kirtis-v2")
CORPUS_DIR = HERE / "lexicon"       # forms + homographs extracted from meldynamics/lt-stressed-corpus (CC BY 4.0)
AMB_MAJ = 1.01   # >1: the model decides every ambiguous form (constrained to attested variants); 0.9 was right for v12
USE_RULES = False   # old RuleLayer (overrides / svogunas lexicon / suffix rules) between the corpus lexicon and the model
VOW = set("aeiouyąęėįųū")
COMB = {"short": "̀", "acute": "́", "tilde": "̃"}
MARKS = set(COMB.values())
WORD = re.compile(r"[\w'̀́̃]+|[^\w\s]", re.UNICODE)

_tok = _model = _rules = None
_clex_raw = json.load(open(CORPUS_DIR / "corpus_lexicon.json", encoding="utf-8"))
CLEX = {k: v["m"] for k, v in _clex_raw.items()}
CLEX_RARE = {k for k, v in _clex_raw.items() if v["n"] <= 1}
# second lexicon: en.wiktionary Lithuanian (kaikki.org extract, CC BY-SA), 139k stressed inflected forms incl. 2.5k names;
# 96 % agreement with the corpus on shared forms (the rest are homographs) — used only for forms the corpus does not know
WLEX = {k: next(iter(v)) for k, v in json.load(open(Path(__file__).parent / "wikt_lexicon.json", encoding="utf-8")).items() if len(v) == 1}   # single sightings: not authoritative for capitalised words (Nausėdà was a 1990s common noun)
CAMB = json.load(open(CORPUS_DIR / "corpus_ambiguous.json", encoding="utf-8"))
CBARE = set(json.load(open(CORPUS_DIR / "corpus_bare.json", encoding="utf-8")))   # corpus leaves these unmarked (a), pan., NATO


def _load():
    global _tok, _model, _rules
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).eval()
        _rules = RuleLayer()


def mark_word(word, syl, acc):
    """Render (syllable index, accent) as a combining mark. Started as space/app.py's
    renderer; re-fitted to the corpus 2026-09-06 (round trip 99.66% -> 99.95% of
    1.16M tokens): iū glide, tilde on the sonorant of a mixed diphthong (klieñtas,
    priim̃ti) except native ie/uo+sonorant (kasdiẽn, dviẽm), grave on the second
    vowel of a diphthong (Ukraìna)."""
    groups = nucleus_groups(word)
    if not groups or syl >= len(groups):
        return word
    g = groups[syl]; w = word.lower()
    vowels = [i for i in g if w[i] in VOW]
    if (len(vowels) >= 2 and w[vowels[0]] == "i" and w[vowels[1]] in GLIDE_VOW
            and (vowels[0] == 0 or w[vowels[0] - 1] not in VOW)):
        vowels = vowels[1:]
    if acc == "tilde" and len(g) > 1:
        if w[g[-1]] in SON:
            after = w[g[-1] + 1] if g[-1] + 1 < len(w) else ""
            if (len(vowels) >= 2 and w[vowels[0]] + w[vowels[1]] in ("ie", "uo")
                    and not (w[vowels[0]] == "i" and after == "t")):
                pos = vowels[1]
            else:
                pos = g[-1]
        else:
            pos = vowels[1] if len(vowels) >= 2 else g[-1]
    elif acc == "short" and len(vowels) >= 2:
        pos = vowels[1]
    else:
        pos = vowels[0] if vowels else g[0]
    return word[:pos + 1] + COMB[acc] + word[pos + 1:]


def recase(marked, orig):
    """apply orig's casing to a lowercase marked form"""
    bases = [c for c in unicodedata.normalize("NFD", orig) if unicodedata.category(c) != "Mn"]
    out, oi = [], 0
    for ch in unicodedata.normalize("NFD", marked):
        if unicodedata.category(ch) == "Mn":
            out.append(ch); continue
        out.append(ch.upper() if oi < len(bases) and bases[oi].isupper() else ch); oi += 1
    return unicodedata.normalize("NFC", "".join(out))


# Optional first-name table with VLKK stress and declension (vardai.vlkk.lt). NOT shipped in this repo:
# the state language commission's data is not ours to redistribute. Without it, first names are stressed by
# the model like any other out-of-vocabulary word, which is right most of the time and wrong for the names
# whose stress is not predictable from spelling. Build your own as {"Jonas": {"m": "Jõnas"}, ...} and drop it
# in text/names/names_first_vlkk.json.
_names_f = HERE / "names" / "names_first_vlkk.json"
NAMES = {k: v["m"] for k, v in json.load(open(_names_f, encoding="utf-8")).items()} if _names_f.exists() else {}
_ov_raw = {unicodedata.normalize("NFC", k): unicodedata.normalize("NFC", v) for k, v in json.load(open(HERE / "stress_overrides.json", encoding="utf-8")).items() if not k.startswith("_")}
OVERRIDES = {k: v for k, v in _ov_raw.items() if not k[:1].isupper()}
OVERRIDES_CAP = {k.lower(): v for k, v in _ov_raw.items() if k[:1].isupper()}     # apply only to capitalised tokens (proper nouns)
FOREIGN = {l.strip().lower() for l in open(HERE / "foreign_bare.txt", encoding="utf-8") if l.strip() and not l.startswith("#")}
LT_LETTERS = set("aąbcčdeęėfghiįyjklmnoprsštuųūvzž")
FOREIGN_SHAPE = re.compile(r"(oo|ee|th|ph|ck|sh|ch|ee|ay|ey|oy)|[bcdfgklmnprtvz]$|y$")


def foreign(w):
    """a word the stresser must not mark: listed brand names, non-Lithuanian letters (w, x, q, ö...), or a capitalised
    word the corpus does not know that looks English (final consonant other than s, -y, oo/ee/th/ck...)"""
    lw = unicodedata.normalize("NFC", w.lower())
    if lw in FOREIGN: return True
    letters = [c for c in lw if c.isalpha()]
    if any(c not in LT_LETTERS for c in letters): return True
    if surname_rule(w): return False                                   # Savickas has "ck" but is a Lithuanian surname
    return w[:1].isupper() and lw not in CLEX and bool(FOREIGN_SHAPE.search(lw))


def passthrough(w):
    """single letters (list labels), digits, ALL-CAPS acronyms, foreign names — unless the corpus knows the form (o, į, ...)"""
    letters = [c for c in w if c.isalpha()]
    if unicodedata.normalize("NFC", w.lower()) in CLEX and len(letters) <= 1:
        return False
    # ALL-CAPS up to 4 letters = acronym (JAV, NATO, ES); longer = a word set in caps (PIRMININKAS)
    return (len(letters) <= 1 or any(c.isdigit() for c in w)
            or (2 <= len(letters) <= 4 and all(c.isupper() for c in letters)) or foreign(w))


def stress_words(words, initial=None):
    """[(marked_word, source, confidence)] for a list of word tokens (one sentence).
    initial: indices of sentence-initial words — capitalised-only overrides (proper nouns) do not apply there,
    because a sentence-initial 'Senos' is just as likely the adjective."""
    initial = set(initial or ())
    _load()
    enc = _tok([words], is_split_into_words=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**enc).logits[0]
    probs = torch.softmax(logits, -1); preds = logits.argmax(-1)
    first = {}
    for pos, wid in enumerate(enc.word_ids(0)):
        if wid is not None and wid not in first:
            first[wid] = pos

    def model_pred(wi, w):
        if wi not in first:
            return None
        p = int(preds[first[wi]]); conf = float(probs[first[wi]][p])
        if p == 0:
            return (None, None, conf)
        return ((p - 1) // 3, ["short", "acute", "tilde"][(p - 1) % 3], conf)

    out = []
    for wi, w in enumerate(words):
        lw = unicodedata.normalize("NFC", w.lower())
        if passthrough(w) or lw in CBARE:
            out.append((w, "pass", 1.0)); continue
        if lw in OVERRIDES or (w[:1].isupper() and wi not in initial and lw in OVERRIDES_CAP):
            out.append((recase(OVERRIDES.get(lw) or OVERRIDES_CAP[lw], w), "override", 1.0)); continue
        if w[:1].isupper() and lw in NAMES:
            out.append((recase(NAMES[lw], w), "vlkk", 1.0)); continue
        if lw in CLEX and not (w[:1].isupper() and lw in CLEX_RARE):
            out.append((recase(CLEX[lw], w), "clex", 1.0)); continue
        if lw not in CLEX and lw not in CAMB and lw in WLEX:
            out.append((recase(WLEX[lw], w), "wikt", 1.0)); continue
        if lw in CAMB:
            var = CAMB[lw]; pick = max(var, key=var.get); src = "camb-major"
            if var[pick] / sum(var.values()) < AMB_MAJ:
                mp = model_pred(wi, w)
                if mp and mp[0] is not None:
                    cand = unicodedata.normalize("NFC", mark_word(lw, mp[0], mp[1]))
                    if cand in var:
                        pick, src = cand, "camb-model"
            out.append((recase(pick, w), src, 1.0)); continue
        hit = _rules.lookup(w) if USE_RULES else None
        if hit:
            syl, acc, src = hit
            out.append((mark_word(w, syl, acc), src.split(":")[0], 1.0)); continue
        sr = surname_rule(w)
        if sr:
            out.append((sr, "surname-rule", 1.0)); continue
        # compounds with an international prefix keep the base word's stress: auto+magistrãlė -> automagistrãlė
        cp = compound_stress(w)
        if cp:
            out.append((cp, "compound", 1.0)); continue
        mp = model_pred(wi, w)
        if mp is None:
            out.append((w, "trunc", 0.0)); continue
        syl, acc, conf = mp
        if syl is None:
            out.append((w, "model0", conf)); continue
        if conf < BARE_BELOW:                   # a guessed mark is worse than none: the voice stresses bare words itself fairly well (KJ, 2026-09-07)
            out.append((w, "model-bare", conf)); continue
        out.append((mark_word(w, syl, acc), "model", conf))
    return out


BARE_BELOW = 0.80   # 3k held-out: model guesses in the 0.6–0.8 band are right only ~2/3 of the time; the voice reads bare words better than that
PREFIXES = ("auto", "euro", "foto", "video", "tele", "radio", "kino", "moto", "elektro", "mikro", "makro", "super", "hiper", "ultra",
            "mega", "kiber", "eko", "bio", "geo", "agro", "aero", "anti", "pseudo", "neo", "termo", "hidro", "turbo", "infra", "inter", "multi")


# surname suffixes with fixed stress, all cases: (regex on the bare lowercase form, replacement with the mark)
SURNAME_RULES = [
    (r"^(.+)([ae])vič(ius|iaus|iui|ių|iumi|iau|ienė|ienės|ienei|ienę|iene|iūtė|iūtės|iūtei|iūtę|iūte)$", lambda m: f"{m.group(1)}{m.group(2)}\u0303vič{m.group(3)}"),   # Sinkẽvičius, Aleksandrãvičiaus
    (r"^(.+)aitė(|s)$", lambda m: f"{m.group(1)}a\u0301itė{m.group(2)}"), (r"^(.+)ait(ei|ę|e)$", lambda m: f"{m.group(1)}a\u0301it{m.group(2)}"),          # Grybauskáitė
    (r"^(.+)ausk(as|o|ui|ą|u|e|ienė|ienės|ienei|ienę|iene|aitė|aitės|aitei|aitę|aite)$", lambda m: f"{m.group(1)}a\u0301usk{m.group(2)}"),          # Kavaliáuskas, Grybauskáitė handled below
    (r"^(.+)insk(as|o|ui|ą|u|e|ienė|ienės|ienei|ienę|iene)$", lambda m: f"{m.group(1)}i\u0300nsk{m.group(2)}"),                                  # Širìnskas
    (r"^(.+)ick(as|o|ui|ą|u|e|ienė|ienės|ienei|ienę|iene)$", lambda m: f"{m.group(1)}i\u0300ck{m.group(2)}"),                                    # Savìckas, Savìcko
    (r"^(.+)eck(as|o|ui|ą|u|e)$", lambda m: f"{m.group(1)}e\u0300ck{m.group(2)}"),
    (r"^(.+)ūn(as|o|ui|ą|u|e|ienė|ienės|ienei|ienę|iene)$", lambda m: f"{m.group(1)}ū\u0301n{m.group(2)}"),                                      # Valančiū́nas
    (r"^(.+)ėn(as|o|ui|ą|u|e|ienė|ienės|ienei|ienę|iene)$", lambda m: f"{m.group(1)}ė\u0301n{m.group(2)}"),
    (r"^(.+)yn(as|o|ui|ą|u|e|ienė|ienės|ienei|ienę|iene)$", lambda m: f"{m.group(1)}y\u0301n{m.group(2)}"),                                      # Katelýnas (like laivýnas)
    (r"^(.+)yt(ė|ės|ei|ę|e|ėje)$", lambda m: f"{m.group(1)}y\u0301t{m.group(2)}"),                                                              # Šimonýtė, Sabonýtė (feminine -ytė: stress on ý)
    (r"^(.+)on(is|io|iui|į|iu|yje|ienė|ienės|ienei|ienę|iene)$", lambda m: f"{m.group(1)}o\u0303n{m.group(2)}"),                                # Sabõnis
    (r"^(.+)ait(is|į|yje)$", lambda m: f"{m.group(1)}a\u0301it{m.group(2)}"), (r"^(.+)aič(io|iui|iu)$", lambda m: f"{m.group(1)}a\u0301ič{m.group(2)}"),   # Žemaitáitis, Žemaitáičio
    (r"^(.+)el(is|io|iui|į|iu|yje)$", lambda m: f"{m.group(1)}e\u0303l{m.group(2)}"),                                                            # Skvernẽlis
    (r"^(.+)ien(ė|ės|ei|ę|e|ėje)$", lambda m: f"{m.group(1)}ie\u0303n{m.group(2)}"),                                                             # Ruginiẽnė (last resort: -iẽnė when no suffix rule above applied)
]


def surname_rule(w):
    """capitalised word with a productive surname suffix -> stressed form (fixed-stress suffixes only); else None"""
    if not w[:1].isupper(): return None
    lw = unicodedata.normalize("NFC", w.lower())
    for pat, rep in SURNAME_RULES:
        m = re.match(pat, lw)
        if m and len(m.group(1)) >= 2:
            return recase(unicodedata.normalize("NFC", rep(m)), w)
    return None


def compound_stress(w):
    """prefix + a base word -> prefix unmarked + base stressed on its own (lexicon, or the model on the base alone
    when it is confident); the whole-word model guess for such compounds is often wrong (automagĩstralėje)"""
    lw = unicodedata.normalize("NFC", w.lower())
    for pre in PREFIXES:
        if lw.startswith(pre) and len(lw) - len(pre) >= 5:
            rest = w[len(pre):]
            marked, src, conf = stress_words([rest])[0]
            if src in ("clex", "camb-major", "camb-model", "override") or (src == "model" and conf >= 0.9):
                return w[:len(pre)] + marked
    return None


def stress_text(text, with_sources=False, normalize=True):
    """mark every word token in place; whitespace and punctuation are kept verbatim.
    normalize=True first spells out numbers, units, dates, times (normalize_lt) so no digit reaches the voice."""
    if normalize:
        from normalize_lt import normalize as _norm
        text = _norm(text)
    spans = [m for m in WORD.finditer(text) if re.match(r"[\w']", m.group())]
    initial = {i for i, m in enumerate(spans) if i == 0 or re.search(r"[.!?…]\s*[„“\"(]*\s*$", text[spans[i - 1].end(): m.start()] and text[:m.start()][-6:] or "")}
    initial = {i for i, m in enumerate(spans) if i == 0 or re.search(r"[.!?…][\s„“\"()–-]*$", text[:m.start()])}
    res = stress_words([m.group() for m in spans], initial=initial)
    out, last = [], 0
    for m, (marked, _, _) in zip(spans, res):
        out.append(text[last:m.start()]); out.append(marked); last = m.end()
    out.append(text[last:])
    marked = "".join(out)
    return (marked, res) if with_sources else marked


if __name__ == "__main__":
    for s in sys.argv[1:]:
        m, res = stress_text(s, with_sources=True)
        print(m)
        low = [f"{w}({src},{c:.2f})" for w, src, c in res if src == "model0" or c < 0.95]
        if low:
            print("   unsure:", " ".join(low))
