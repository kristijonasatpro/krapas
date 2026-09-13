#!/usr/bin/env python3
"""Pick a reading script from the VDU stressed corpus: sentences that already carry human stress marks,
so the stress is right before you open your mouth.

    python3 build_script.py --n 600 --out script.json          # downloads the corpus on first run
    python3 build_script.py --n 600 --corpus /path/train.parquet

Writes <out>.json (id, src, plain, stressed, words, n) and <out>.txt (the numbered list you read from).

What it filters out, and why each rule exists:
  * rows where the corpus itself was unsure (fallback engine, unresolved words, several equally good
    stressings) — the stress has to be gold, or the model learns someone's guess;
  * digits, Roman numerals, abbreviations — you would have to decide how to read them out loud, and a
    reading script is the wrong place for that decision (see the abbreviation set in script-b);
  * ALL-CAPS tokens, brackets, dash chains, unbalanced quotes, dialogue attributions — they break
    intonation or make you hesitate;
  * violent / graphic subject matter — you are going to read this out loud for two hours.

Quotas (the defaults are what we recorded): genre fiction 45 % / press 40 % / parliament 15 %;
length short (<=5 words) 15 % / mid 75 % / long (>=16) 10 %; endings question 15 % / exclamation 7 %.
Short sentences with clean endings and real questions are what the model lacks most, so they are
over-represented relative to how often they occur in the corpus.
"""
import argparse, collections, json, random, re, unicodedata as ud
import pyarrow.parquet as pq

OK_CHARS = re.compile(r"^[A-Za-zĄČĘĖĮŠŲŪŽąčęėįšųūž ,.;:!?\-–—„“”\"'()]+$")
ABBR = re.compile(r"\b[a-ząčęėįšųūž]{1,4}\.(?!$)|\b(proc|pvz|t\. y|tūkst|mln|mlrd|kt|pan|val|min|sek|str|dr|prof|doc|akad|inž|ir kt)\.", re.I)
BAD = re.compile(r"nužud|žudik|lavon|išprievart|prievarta|kraujuo|savižud|pakor|sušaud|išžag|homoseks|pedofil|nacių|holokaust", re.I)
CAPS = re.compile(r"\b[A-ZĄČĘĖĮŠŲŪŽ]{2,}\b")
GENRES = {"gro": .45, "pub": .40, "ste": .15}      # fiction / press / parliament transcripts
LENGTHS = {"short": .15, "mid": .75, "long": .10}
ENDINGS = {"q": .15, "x": .07, "s": .78}
QUALITY = ["digit_token_count", "roman_numeral_token_count", "multiple_best_stress_option_word_count",
           "phonology_engine_word_count", "lki_words_word_count", "unresolved_word_count", "no_stress_option_word_count"]


def clean(s):
    s = re.sub(r"^[-–—]\s*", "", s.strip())
    return re.sub(r"\s+", " ", re.sub(r"\s+([,.;:!?])", r"\1", s))


def usable(t):
    if not OK_CHARS.match(t) or not re.search(r"[.!?]$", t) or not re.match(r"^[A-ZĄČĘĖĮŠŲŪŽ„\"(]", t): return False
    if ABBR.search(t) or BAD.search(t) or CAPS.search(t) or "(" in t or ")" in t: return False
    if "..." in t or "?!" in t or "?.." in t: return False
    if re.search(r",\s*[-–—]|[”“\"!?],?\s*[-–—]|\w[–—]\w|\s[-–—]\s\w+\s[-–—]|(\w+)-\1", t, re.I): return False
    if t.count('"') % 2 or t.count("„") != t.count("“") + t.count("”"): return False
    return True


def corpus_path(given):
    if given: return given
    from huggingface_hub import hf_hub_download
    return hf_hub_download("meldynamics/lt-stressed-corpus", "data/default/train.parquet", repo_type="dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="how many sentences (600 = about an hour of reading)")
    ap.add_argument("--out", default="script", help="output basename")
    ap.add_argument("--corpus", default=None, help="train.parquet; downloaded from the Hub when omitted")
    ap.add_argument("--seed", type=int, default=20260907)
    a = ap.parse_args()
    random.seed(a.seed)

    cols = ["source_file", "sent_id", "original_text", "stressed_text_unicode"] + QUALITY
    rows = pq.read_table(corpus_path(a.corpus), columns=cols).to_pylist()
    pool, seen = collections.defaultdict(list), set()
    for r in rows:
        if any(r[k] for k in QUALITY): continue
        genre = r["source_file"].split("/")[-1][:3]
        if genre not in GENRES: continue
        t, st = clean(r["original_text"]), clean(r["stressed_text_unicode"])
        if not usable(t) or t.lower() in seen: continue
        n = len(t.split())
        if not 2 <= n <= 25: continue
        seen.add(t.lower())
        bucket = "short" if n <= 5 else ("long" if n >= 16 else "mid")
        end = "q" if t.endswith("?") else ("x" if t.endswith("!") else "s")
        pool[(genre, bucket, end)].append(dict(id=r["sent_id"], src=genre, plain=t,
                                              stressed=ud.normalize("NFC", st), words=n))

    target = {(g, b, e): a.n * gw * bw * ew for g, gw in GENRES.items()
              for b, bw in LENGTHS.items() for e, ew in ENDINGS.items()}
    picked = []
    for k, want in sorted(target.items(), key=lambda kv: -kv[1]):
        c = pool.get(k, []); random.shuffle(c); picked += c[: round(want)]
    fill = [s for c in pool.values() for s in c if s not in picked]      # top up cells the corpus could not fill
    random.shuffle(fill)
    script = (picked + fill[: max(0, a.n - len(picked))])[: a.n]
    random.shuffle(script)

    # warm-up: the first 20 are plain mid-length statements, so the first minutes are not the hard ones
    warm = [x for x in script if 6 <= x["words"] <= 12 and x["plain"].endswith(".")][:20]
    script = warm + [x for x in script if x not in warm]
    for i, s in enumerate(script, 1): s["n"] = i

    json.dump(script, open(f"{a.out}.json", "w"), ensure_ascii=False, indent=0)
    open(f"{a.out}.txt", "w").write("\n".join(f"{s['n']:03d}. {s['plain']}" for s in script) + "\n")
    c = collections.Counter((s["src"], "short" if s["words"] <= 5 else "long" if s["words"] >= 16 else "mid") for s in script)
    print(len(script), "sentences,", sum(s["words"] for s in script), "words")
    print("buckets:", dict(c))
    print("endings:", dict(collections.Counter(s["plain"][-1] for s in script)))


if __name__ == "__main__":
    main()
