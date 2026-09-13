#!/usr/bin/env python3
"""Transcribe every take with a local Lithuanian ASR model, so the next step knows what you actually said
and where the speech starts and ends.

    python3 asr_check.py takes/ --out takes.asr.json
    python3 asr_check.py takes/ --model kristijonas/paprika-whisper-lt-v3 --device mps

One file in, one JSON out: {"001": {"text": ..., "start": ..., "end": ...}, ...}. Times are seconds inside
that take. Any model that returns segment timestamps works — Paprika V3 (whisper-large-v3-turbo fine-tuned
for Lithuanian) is free and local, which is the whole point; a paid API only saves you time.
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("takes", help="directory of NNN.wav takes")
    ap.add_argument("--out", default=None, help="default <takes>.asr.json")
    ap.add_argument("--model", default="kristijonas/paprika-whisper-lt-v3")
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (auto)")
    a = ap.parse_args()
    import torch
    from transformers import pipeline
    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    asr = pipeline("automatic-speech-recognition", model=a.model, device=dev,
                   dtype=torch.float32, return_timestamps=True)
    takes = sorted(Path(a.takes).glob("[0-9][0-9][0-9].wav"))
    out, path = {}, Path(a.out or f"{a.takes.rstrip('/')}.asr.json")
    done = json.load(open(path)) if path.exists() else {}          # resume: re-running skips what is already there
    for i, f in enumerate(takes, 1):
        if f.stem in done:
            out[f.stem] = done[f.stem]; continue
        r = asr(str(f), generate_kwargs={"language": "lt", "task": "transcribe"})
        ch = [c for c in r.get("chunks") or [] if c.get("timestamp")]
        start = min((c["timestamp"][0] for c in ch if c["timestamp"][0] is not None), default=None)
        end = max((c["timestamp"][1] for c in ch if c["timestamp"][1] is not None), default=None)
        out[f.stem] = dict(text=r["text"].strip(), start=start, end=end)
        if i % 10 == 0 or i == len(takes):
            json.dump(out, open(path, "w"), ensure_ascii=False, indent=0)
            print(f"{i}/{len(takes)}  {f.stem}: {out[f.stem]['text'][:70]}", flush=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=0)
    print("wrote", path, len(out), "takes")


if __name__ == "__main__":
    main()
