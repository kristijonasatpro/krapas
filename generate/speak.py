#!/usr/bin/env python3
"""Read a text file out loud with a fine-tuned OmniVoice voice: normalise the numbers, mark the stress,
generate one sentence at a time, join with pauses.

    python3 speak.py text.txt out/                                  # the published Regina voice
    python3 speak.py text.txt out/ --ckpt ../train/results/checkpoint-3000
    python3 speak.py text.txt out/ --plain                          # no stress marks, to hear the difference

Writes out/pNsM.wav per sentence, out/full.wav, and out/stressed.json — the three stages of every sentence
(as written, normalised, stressed), which is what you read when something sounds wrong: nine times out of
ten the fault is visible in the text and is a one-word dictionary fix, not a retrain.

Sentence by sentence is not laziness. The training clips are a few seconds long, so a single long request
drifts; 0.4 s between sentences and 0.9 s between paragraphs is what a reader does anyway.
"""
import argparse, glob, json, os, re, sys, time, unicodedata as ud
from pathlib import Path
import numpy as np, soundfile as sf, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("KRAPAS_TEXT", str(HERE.parent / "text")))
import kirtis_v2
from normalize_lt import normalize
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
import omnivoice.models.omnivoice as _ov

# A model trained with a free-form speaker tag ("Speaker: Regina") needs the tag passed through: the stock
# resolver only understands gender/accent words and lower-cases whatever it is given.
_ov._resolve_instruct = lambda instruct, *a, **k: instruct

MARKS = "̀́̃"
SR, GAP_SENT, GAP_PARA = 24000, 0.4, 0.9


def train_form(s):
    """stress marks as separate combining code points, the rest composed — the form the model was trained on"""
    return "".join(ud.normalize("NFC", seg) if seg not in MARKS else seg
                   for seg in re.split(f"([{MARKS}])", ud.normalize("NFD", s)))


def sentences(par):
    """split at sentence punctuation; a single capital letter + '.' is an initial, not a sentence end"""
    return [s for s in re.split(r"(?<![A-ZĄČĘĖĮŠŲŪŽ]\.)(?<=[.!?“”\"])\s+(?=[„“\"A-ZĄČĘĖĮŠŲŪŽ])", par.strip()) if s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt"); ap.add_argument("out")
    ap.add_argument("--ckpt", default="kristijonas/krapas-Regina")
    ap.add_argument("--guidance", type=float, default=4.0, help="4 follows the stress marks; the library default of 2 ignores some")
    ap.add_argument("--num-step", type=int, default=32, help="16 is twice as fast and noisier")
    ap.add_argument("--instruct", default=None, help='speaker tag for a multi-voice checkpoint, e.g. "Speaker: Regina"')
    ap.add_argument("--plain", action="store_true", help="normalise numbers but add no stress marks")
    ap.add_argument("--initials", default="letter", choices=["letter", "drop", "keep"],
                    help="'G. Nausėda' -> 'gė Nausėda' (letter), 'Nausėda' (drop), or left alone")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    text = open(a.txt, encoding="utf-8").read()
    text = re.sub(r"\S+@\S+\.\w+", lambda m: m.group().replace("@", " eta ").replace(".", " taškas "), text)
    plan = []
    for pi, p in enumerate([p for p in re.split(r"\n\s*\n", text) if p.strip()]):
        orig = sentences(p)
        norm = sentences(normalize(p, initials=a.initials))     # normalise the paragraph: abbreviations span sentence marks
        if len(orig) != len(norm): orig = norm                  # the split moved; the normalised text wins
        for si, (s, n) in enumerate(zip(orig, norm)):
            marked = n if a.plain else kirtis_v2.stress_text(n, normalize=False)
            plan.append({"p": pi, "i": si, "plain": s, "normalized": n, "stressed": train_form(marked)})
    json.dump(plan, open(out / "stressed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(plan), "sentences", flush=True)

    # a checkpoint fetched from a training run has no audio tokenizer of its own; borrow the base model's
    if os.path.isdir(a.ckpt) and not os.path.isdir(f"{a.ckpt}/audio_tokenizer"):
        base = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/*/audio_tokenizer"))
        if not base:
            sys.exit("no audio_tokenizer: run once with --ckpt k2-fsa/OmniVoice to cache the base model, or copy it in")
        os.symlink(base[0], f"{a.ckpt}/audio_tokenizer")

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = OmniVoice.from_pretrained(a.ckpt, device_map=dev, dtype=torch.float32)
    cfg = OmniVoiceGenerationConfig(guidance_scale=a.guidance, num_step=a.num_step)

    full, cur = [], -1
    for row in plan:
        if row["p"] != cur and cur >= 0: full.append(np.zeros(int(GAP_PARA * SR), np.float32))
        cur = row["p"]; t0 = time.time()
        w = np.asarray(model.generate(text=row["stressed"], language="lt", instruct=a.instruct,
                                      generation_config=cfg, normalize_text=False)[0], np.float32)
        name = f"p{row['p']+1}s{row['i']+1}.wav"
        sf.write(out / name, w, SR); row["dur"] = round(len(w) / SR, 2)
        print(f"{name}: {len(w)/SR:.1f}s in {time.time()-t0:.1f}s | {row['normalized'][:70]}", flush=True)
        full += [w, np.zeros(int(GAP_SENT * SR), np.float32)]
    sf.write(out / "full.wav", np.concatenate(full), SR)
    json.dump(plan, open(out / "stressed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("full.wav", round(sum(r["dur"] for r in plan), 1), "s")


if __name__ == "__main__":
    main()
