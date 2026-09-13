#!/usr/bin/env python3
"""Transcribe every take with a Lithuanian ASR model so the next step can tell whether you read what is
written. Runs on your own machine; no API, no cost.

    python3 asr_check.py takes --out takes.asr.json
    python3 asr_check.py takes --model kristijonas/paprika-whisper-lt-v3 --device mps

One take in, one line out: {"001": {"text": "..."}, ...}. Re-running skips the takes already in the file,
so an interrupted pass costs nothing.

The default is Kmynas v4 (parakeet-tdt fine-tuned for Lithuanian). Any Hugging Face ASR model works — whisper
family included — because all this step needs is the words; the clip boundaries come from the audio itself
in build_clips.py, not from the model.
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", help="directory of NNN.wav takes")
    ap.add_argument("--out", default=None, help="default <takes>.asr.json")
    ap.add_argument("--model", default="kristijonas/kmynas-parakeet-lt-v4")
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (auto)")
    a = ap.parse_args()

    import torch
    from transformers import pipeline
    dev = a.device or ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    asr = pipeline("automatic-speech-recognition", model=a.model, device=dev, dtype=torch.float32)
    # whisper needs to be told which language it is hearing; transducers do not take the argument at all
    kw = {"generate_kwargs": {"language": "lt", "task": "transcribe"}} \
        if getattr(asr.model.config, "model_type", "") == "whisper" else {}

    takes = sorted(Path(a.takes).glob("[0-9][0-9][0-9].wav"))
    path = Path(a.out or f"{a.takes.rstrip('/')}.asr.json")
    out = json.load(open(path)) if path.exists() else {}
    todo = [f for f in takes if f.stem not in out]
    print(f"{len(takes)} takes, {len(todo)} to transcribe, model {a.model} on {dev}", flush=True)
    for i, f in enumerate(todo, 1):
        out[f.stem] = {"text": asr(str(f), **kw)["text"].strip()}
        if i % 10 == 0 or i == len(todo):
            json.dump(out, open(path, "w"), ensure_ascii=False, indent=0)
            print(f"{i}/{len(todo)}  {f.stem}: {out[f.stem]['text'][:70]}", flush=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=0)
    print("wrote", path, len(out), "takes")


if __name__ == "__main__":
    main()
