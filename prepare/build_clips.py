#!/usr/bin/env python3
"""Turn the recorded takes into training clips: check that you read what is written, cut the breath off
both ends, and write the manifests the trainer eats.

    python3 build_clips.py --script ../script/script-a-567.json --takes takes --asr takes.asr.json --out clips

Per take: compare the transcript with the script (word and character error rate) and drop the ones where
you read something else; find the sentence in the take by energy and cut to it; resample to 24 kHz mono;
level every clip to the same loudness. Writes clips/wav/*.wav, train.jsonl + dev.jsonl (what the trainer
reads), all_meta.json, review.json (takes to listen to again) and stats.json.

Finding the sentence by energy rather than by ASR word times is deliberate. A word-level timestamp from a
whisper-family model on a 6-second file is usually just "0 to 6 seconds", which is no help, and the thing
you actually need to cut is the breath — which no ASR marks. So: frames above THRESH_DB below the take's
peak are speech, runs of speech separated by more than GAP_S of silence are separate events, and the sentence spans the
first to the last run that holds at least MIN_EVENT_FRAC of the longest run's speech. A breath, a lip smack
or a false start is short and drops out; a phrase after a comma is not. This is why the recording
instruction is "sentence, then a pause, and only then breathe": the pause is what makes the breath a
separate run.

Checked against 60 takes that had also been cut using word timestamps from a commercial ASR: the clip
boundaries agree to 58 ms on average, and where they differ this path keeps slightly more silence rather
than clipping a consonant (worst case +260 ms).
"""
import argparse, json, re, unicodedata as ud
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

FRAME_S, GAP_S, THRESH_DB = 0.010, 0.30, -35.0
MIN_EVENT_FRAC = 0.25      # a run shorter than this fraction of the sentence body is a breath or a click
PAD_IN, PAD_OUT = 0.080, 0.120          # hard cap kept beyond the sentence on each side
FADE_S, TARGET_DBFS, SR_OUT = 0.020, -20.0, 24000
MARKS = "̀́̃"


def train_form(s):
    """stress marks as separate combining code points, the rest composed — the form the model was trained on"""
    return "".join(ud.normalize("NFC", seg) if seg not in MARKS else seg
                   for seg in re.split(f"([{MARKS}])", ud.normalize("NFD", s)))


def bare(s):
    return "".join(c for c in ud.normalize("NFD", s.lower()) if not ud.combining(c))


def words(s): return re.findall(r"[a-ząčęėįšųūž]+", bare(s))


def letters(s): return re.sub(r"[^a-ząčęėįšųūž]", "", bare(s))


def edit_rate(ref, hyp):
    """Levenshtein rate over whatever units are handed in (words or characters)"""
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(1, len(ref))


def sentence_span(x, sr):
    """(start, end) of the longest run of speech, in seconds, or None for a silent take"""
    fl = max(1, int(FRAME_S * sr))
    n = len(x) // fl
    if n == 0: return None
    rms = np.sqrt((x[: n * fl].reshape(n, fl) ** 2).mean(1) + 1e-12)
    thr = rms.max() * 10 ** (THRESH_DB / 20)
    loud = rms > thr
    if not loud.any(): return None
    idx = np.flatnonzero(loud)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > int(GAP_S / FRAME_S)) + 1)
    body = max(len(r) for r in runs)
    keep = [r for r in runs if len(r) >= max(MIN_EVENT_FRAC * body, 8)]   # drop breaths, keep phrases
    return keep[0][0] * FRAME_S, (keep[-1][-1] + 1) * FRAME_S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--takes", default="takes")
    ap.add_argument("--asr", default=None, help="asr_check.py output; without it nothing is verified")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--prefix", default="clip")
    ap.add_argument("--pod-dir", default="/workspace/ft/clips/wav", help="where the wavs will live on the training machine")
    ap.add_argument("--max-wer", type=float, default=0.15)
    ap.add_argument("--max-cer", type=float, default=0.04, help="a take passes on either metric: the ASR's spelling is not the truth")
    ap.add_argument("--dev", type=int, default=30, help="held-out clips")
    ap.add_argument("--keep", default="", help="take numbers to keep although the ASR disagrees (you listened)")
    a = ap.parse_args()

    script = {s["n"]: s for s in json.load(open(a.script))}
    asr = json.load(open(a.asr)) if a.asr else {}
    keep = {int(x) for x in a.keep.split(",") if x.strip()}
    out = Path(a.out); (out / "wav").mkdir(parents=True, exist_ok=True)
    rows, review = [], []

    for f in sorted(Path(a.takes).glob("[0-9][0-9][0-9].wav")):
        n = int(f.stem)
        if n not in script:
            review.append(dict(n=n, reason="not in script")); continue
        s = script[n]
        x, sr = sf.read(f, dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        span = sentence_span(x, sr)
        if span is None:
            review.append(dict(n=n, reason="silent take", script=s["plain"])); continue

        heard = (asr.get(f.stem) or {}).get("text", "")
        w = edit_rate(words(s["plain"]), words(heard)) if heard else None
        c = edit_rate(letters(s["plain"]), letters(heard)) if heard else None
        ok = n in keep or w is None or w <= a.max_wer or c <= a.max_cer
        if not ok:
            review.append(dict(n=n, reason="mismatch", wer=round(w, 3), cer=round(c, 3),
                               script=s["plain"], heard=heard)); continue

        t0 = max(0.0, span[0] - PAD_IN); t1 = min(len(x) / sr, span[1] + PAD_OUT)
        seg = x[int(t0 * sr): int(t1 * sr)].copy()
        nf = int(FADE_S * sr); ramp = np.linspace(0, 1, nf, dtype=np.float32)
        seg[:nf] *= ramp; seg[-nf:] *= ramp[::-1]
        y = resample_poly(seg, SR_OUT, sr).astype(np.float32) if sr != SR_OUT else seg
        rms = float(np.sqrt((y ** 2).mean()) + 1e-9)
        y *= min(10 ** (TARGET_DBFS / 20) / rms, 0.98 / max(1e-6, float(np.abs(y).max())))
        cid = f"{a.prefix}_{n:03d}"
        sf.write(out / "wav" / f"{cid}.wav", y, SR_OUT, subtype="PCM_16")
        rows.append(dict(id=cid, n=n, dur=round(len(y) / SR_OUT, 3), plain=s["plain"],
                         text=train_form(s["stressed"]), src=s.get("src", ""),
                         wer=None if w is None else round(w, 3), cer=None if c is None else round(c, 3),
                         heard=heard, trim_in=round(t0, 3), trim_out=round(len(x) / sr - t1, 3)))

    rng = np.random.default_rng(42)
    dev = set(rng.permutation(len(rows))[: a.dev].tolist())
    for name, sel in (("train", [r for i, r in enumerate(rows) if i not in dev]),
                      ("dev", [r for i, r in enumerate(rows) if i in dev])):
        with open(out / f"{name}.jsonl", "w") as fh:
            for r in sel:
                fh.write(json.dumps({"id": r["id"], "audio_path": f"{a.pod_dir}/{r['id']}.wav",
                                     "text": r["text"], "language_id": "lt"}, ensure_ascii=False) + "\n")
    json.dump(rows, open(out / "all_meta.json", "w"), ensure_ascii=False, indent=0)
    json.dump(review, open(out / "review.json", "w"), ensure_ascii=False, indent=1)
    d = np.array([r["dur"] for r in rows]) if rows else np.zeros(1)
    stats = dict(clips=len(rows), dropped=len(review), hours=round(float(d.sum()) / 3600, 3),
                 dur_mean=round(float(d.mean()), 2), dur_max=round(float(d.max()), 2),
                 trimmed_head_s=round(float(sum(r["trim_in"] for r in rows)), 1),
                 trimmed_tail_s=round(float(sum(r["trim_out"] for r in rows)), 1))
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps(stats))
    for r in review[:40]:
        print(f"  {r['n']:03d} {r['reason']}: {r.get('script','')[:60]} || {r.get('heard','')[:60]}")


if __name__ == "__main__":
    main()
