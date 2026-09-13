"""Sentence-by-sentence recorder. Standard library only, runs in your browser.

    python3 rec_server.py ../script/script-a-567.json takes      # then open http://localhost:8765

Shows one stressed sentence at a time; the browser captures raw microphone PCM (all processing off:
no echo cancellation / noise suppression / auto gain) and sends a 16-bit WAV per sentence, which is saved
as takes/NNN.wav (re-recording overwrites). Keys: Space = start/stop, Enter or → = next, ← = back,
P = play the last take. Progress is derived from the files on disk, so restarting is safe.
"""
import json, os, re, sys, time, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "script" / "script-a-567.json"
TAKES = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "takes"; TAKES.mkdir(exist_ok=True)
LETTER = {"a": "a", "ą": "a nosinė", "b": "bė", "c": "cė", "č": "čė", "d": "dė", "e": "e", "ę": "e nosinė", "ė": "ė", "f": "ef", "g": "gė", "h": "ha", "i": "i", "į": "i nosinė",
          "y": "i ilgoji", "j": "jot", "k": "ka", "l": "el", "m": "em", "n": "en", "o": "o", "p": "pė", "r": "er", "s": "es", "š": "eš", "t": "tė", "u": "u", "ų": "u nosinė",
          "ū": "u ilgoji", "v": "vė", "z": "zė", "ž": "žė"}
WORD_ACR = {"NATO", "ELTA", "ESBO", "UNESCO", "LITEXPO"}      # read as words, not letters


def hint(text):
    """how to read initials and acronyms: 'G. -> gė', 'JAV -> jot a vė'"""
    out = []
    for m in re.finditer(r"\b([A-ZĄČĘĖĮŠŲŪŽ])\.(?=\s[A-ZĄČĘĖĮŠŲŪŽ])", text): out.append(f"{m.group(1)}. → {LETTER[m.group(1).lower()]}")
    for m in re.finditer(r"\b([A-ZĄČĘĖĮŠŲŪŽ]{2,6})\b", text):
        a = m.group(1); out.append(f"{a} → {a.lower() if a in WORD_ACR else ' '.join(LETTER[c.lower()] for c in a)}")
    return " · ".join(dict.fromkeys(out))
PORT = int(os.environ.get("REC_PORT", 8765))
LOCK = threading.Lock()


def wav_duration(p: Path):
    try:
        b = p.read_bytes()
        if len(b) < 44: return 0.0
        sr = int.from_bytes(b[24:28], "little"); ch = int.from_bytes(b[22:24], "little"); bps = int.from_bytes(b[34:36], "little")
        return (len(b) - 44) / max(1, sr * ch * bps // 8)
    except Exception:
        return 0.0


def takes_state():
    out = {}
    for p in TAKES.glob("*.wav"):
        try: out[int(p.stem)] = round(wav_duration(p), 2)
        except ValueError: pass
    return out


PAGE = r"""<!doctype html><html lang="lt"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Įrašymas</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>
:root{--bg:#eceef1;--card:#fff;--ink:#161a20;--mut:#6a727d;--line:#d9dde3;--btn:#f2f4f7;--on:#0e6a66;--onink:#fff;--rec:#c8372d;--ok:#2f8f5b;--meter:#7fb8a8}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1a1e24;--ink:#e9ecef;--mut:#98a0aa;--line:#2b3138;--btn:#252a31;--on:#2fa39b;--onink:#08211f;--rec:#e0564b;--ok:#4fbf82;--meter:#2fa39b}}
*{box-sizing:border-box} html,body{height:100%}
body{background:var(--bg);color:var(--ink);font:16px/1.45 -apple-system,system-ui,"Segoe UI",Roboto,sans-serif;margin:0;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font:600 16px/1.2 -apple-system,system-ui,sans-serif;margin:0} .sp{flex:1}
.prog{font-variant-numeric:tabular-nums;color:var(--mut)} select{font:14px system-ui;max-width:280px;background:var(--btn);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px}
main{flex:1;display:grid;place-items:center;padding:24px 16px}
.card{width:min(860px,100%);background:var(--card);border:1px solid var(--line);border-radius:18px;padding:28px 32px;display:grid;gap:18px}
.num{font:600 13px/1 -apple-system,system-ui,sans-serif;letter-spacing:.08em;color:var(--mut);font-variant-numeric:tabular-nums;display:flex;gap:14px;align-items:center}
.state{font-weight:600} .state.rec{color:var(--rec)} .state.ok{color:var(--ok)}
.t{font:400 clamp(26px,3.6vw,40px)/1.4 "Noto Serif",Georgia,"Times New Roman",serif;margin:0;text-wrap:pretty;overflow-wrap:anywhere}
.meter{height:8px;background:var(--btn);border-radius:4px;overflow:hidden} .meter i{display:block;height:100%;width:0;background:var(--meter);transition:width .06s}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button{font:600 16px/1 -apple-system,system-ui,sans-serif;background:var(--btn);color:var(--ink);border:1px solid var(--line);border-radius:12px;height:52px;padding:0 22px;cursor:pointer}
button.rec{background:var(--rec);color:#fff;border-color:var(--rec)} button.on{background:var(--on);color:var(--onink);border-color:var(--on)} button:disabled{opacity:.4;cursor:default}
button:focus-visible{outline:2px solid var(--on);outline-offset:2px}
.hint{color:var(--mut);font-size:14px} kbd{font:600 12px/1 system-ui;background:var(--btn);border:1px solid var(--line);border-radius:5px;padding:2px 6px}
.err{color:var(--rec);font-weight:600}
</style></head><body>
<header><h1>Įrašymas</h1><span class="prog" id="prog"></span><span class="sp"></span><select id="dev" title="Mikrofonas"></select><button id="arm" onclick="arm()">Įjungti mikrofoną</button></header>
<main><div class="card">
<div class="num"><span id="n">—</span><span class="state" id="state"></span><span id="sr" class="hint"></span></div>
<p class="t" id="t">Spausk „Įjungti mikrofoną“, pasirink savo mikrofoną, tada <kbd>Space</kbd>.</p>
<p class="hint" id="hint"></p>
<div class="meter"><i id="lvl"></i></div>
<div class="row"><button id="prev" onclick="go(-1)">‹ Atgal</button><button id="rec" class="rec" onclick="toggle()" disabled>● Įrašyti</button><button id="play" onclick="play()" disabled>▶ Perklausyti</button><button id="next" class="on" onclick="go(1)">Toliau ›</button><span class="sp"></span><span class="hint" id="msg"></span></div>
<div class="hint"><kbd>Space</kbd> pradėti / sustabdyti · <kbd>→</kbd> arba <kbd>Enter</kbd> kitas · <kbd>←</kbd> atgal · <kbd>P</kbd> perklausyti · pakartojus tą patį numerį įrašas perrašomas</div>
</div></main>
<script>
let S=[],T={},i=0,ctx,node,stream,chunks=[],recording=false,sr=0,lastPeak=0;
const $=id=>document.getElementById(id);
async function load(){const r=await fetch('/script');const d=await r.json();S=d.script;T=d.takes;
 // start at the first sentence without a take
 const first=S.findIndex(s=>!T[s.n]);i=first<0?0:first;render();}
function render(){const s=S[i];$('n').textContent=String(s.n).padStart(3,'0')+' / '+S.length;$('t').innerHTML=s.stressed;$('hint').textContent=s.hint||'';
 const has=T[s.n];$('state').textContent=recording?'● ĮRAŠOMA':(has?'✓ įrašyta '+has.toFixed(1)+' s':'');$('state').className='state '+(recording?'rec':(has?'ok':''));
 $('play').disabled=!has||recording;$('prev').disabled=i===0||recording;$('next').disabled=i===S.length-1||recording;
 const done=Object.keys(T).length;$('prog').textContent=done+' / '+S.length+' įrašyta';}
function go(d){if(recording)return;i=Math.min(S.length-1,Math.max(0,i+d));render();}
async function listDevices(){const ds=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='audioinput');const sel=$('dev');const cur=sel.value;sel.innerHTML='';
 ds.forEach(d=>{const o=document.createElement('option');o.value=d.deviceId;o.textContent=d.label||('Mikrofonas '+(sel.length+1));sel.appendChild(o);});if(cur)sel.value=cur;}
async function arm(){try{
 if(stream){stream.getTracks().forEach(t=>t.stop());}
 const devId=$('dev').value||undefined;
 stream=await navigator.mediaDevices.getUserMedia({audio:{deviceId:devId?{exact:devId}:undefined,channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false,sampleRate:48000}});
 await listDevices();
 const tr=stream.getAudioTracks()[0];const st=tr.getSettings();
 if(ctx)await ctx.close();ctx=new AudioContext({sampleRate:48000});await ctx.resume();sr=ctx.sampleRate;
 const code=`class R extends AudioWorkletProcessor{process(inp){const c=inp[0]&&inp[0][0];if(c)this.port.postMessage(c.slice());return true}}registerProcessor('r',R)`;
 await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code],{type:'application/javascript'})));
 const src=ctx.createMediaStreamSource(stream);node=new AudioWorkletNode(ctx,'r');src.connect(node);
 node.port.onmessage=e=>{const c=e.data;let p=0;for(let k=0;k<c.length;k++){const a=Math.abs(c[k]);if(a>p)p=a;}lastPeak=p;$('lvl').style.width=Math.min(100,Math.round(p*140))+'%';if(recording)chunks.push(c);};
 $('sr').textContent=sr+' Hz · '+(tr.label||'')+(st.echoCancellation||st.noiseSuppression||st.autoGainControl?' · DĖMESIO: apdorojimas įjungtas':' · be apdorojimo');
 $('rec').disabled=false;$('arm').textContent='Mikrofonas įjungtas';$('msg').textContent='';render();
}catch(e){$('msg').innerHTML='<span class="err">Mikrofonas nepasiekiamas: '+e.message+'</span>';}}
$('dev').onchange=()=>{if(stream)arm();};
function toggle(){if(!ctx)return;if(!recording){chunks=[];recording=true;render();}else{recording=false;render();save();}}
function wav(chs){let n=0;for(const c of chs)n+=c.length;const buf=new ArrayBuffer(44+n*2);const v=new DataView(buf);
 const w=(o,s)=>{for(let k=0;k<s.length;k++)v.setUint8(o+k,s.charCodeAt(k));};
 w(0,'RIFF');v.setUint32(4,36+n*2,true);w(8,'WAVE');w(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);w(36,'data');v.setUint32(40,n*2,true);
 let o=44;for(const c of chs){for(let k=0;k<c.length;k++){const x=Math.max(-1,Math.min(1,c[k]));v.setInt16(o,x<0?x*32768:x*32767,true);o+=2;}}return buf;}
async function save(){const s=S[i];const b=wav(chunks);const dur=(b.byteLength-44)/2/sr;if(dur<0.4){$('msg').textContent='per trumpas ('+dur.toFixed(1)+' s), neišsaugotas';render();return;}
 const r=await fetch('/take/'+s.n,{method:'PUT',body:b});const d=await r.json();T[s.n]=d.dur;$('msg').textContent='išsaugota '+String(s.n).padStart(3,'0')+'.wav ('+d.dur.toFixed(1)+' s)';render();}
function play(){const s=S[i];if(!T[s.n])return;new Audio('/take/'+s.n+'.wav?'+Date.now()).play();}
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;
 if(e.code==='Space'){e.preventDefault();toggle();}else if(e.key==='ArrowRight'||e.key==='Enter'){e.preventDefault();go(1);}else if(e.key==='ArrowLeft'){e.preventDefault();go(-1);}else if(e.key==='p'||e.key==='P'){play();}});
navigator.mediaDevices.addEventListener('devicechange',listDevices);
load();listDevices();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = PAGE.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path == "/script":
            self._json({"script": [{"n": s["n"], "stressed": s["stressed"], "plain": s["plain"], "hint": hint(s["plain"])} for s in json.load(open(SCRIPT))], "takes": takes_state()})
        elif self.path.startswith("/take/") and ".wav" in self.path:
            n = int(self.path.split("/take/")[1].split(".")[0]); p = TAKES / f"{n:03d}.wav"
            if not p.exists(): self.send_response(404); self.end_headers(); return
            b = p.read_bytes(); self.send_response(200); self.send_header("Content-Type", "audio/wav"); self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(404); self.end_headers()

    def do_PUT(self):
        if not self.path.startswith("/take/"): self.send_response(404); self.end_headers(); return
        n = int(self.path.split("/take/")[1]); body = self.rfile.read(int(self.headers["Content-Length"]))
        p = TAKES / f"{n:03d}.wav"; tmp = p.with_suffix(".tmp")
        with LOCK:
            tmp.write_bytes(body); os.replace(tmp, p)
            sent = next((s for s in json.load(open(SCRIPT)) if s["n"] == n), None)
            with open(TAKES / "log.jsonl", "a") as fh:
                fh.write(json.dumps({"n": n, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "dur": round(wav_duration(p), 3), "bytes": len(body),
                                     "plain": sent and sent["plain"], "stressed": sent and sent["stressed"]}, ensure_ascii=False) + "\n")
        self._json({"n": n, "dur": wav_duration(p)})


if __name__ == "__main__":
    if not SCRIPT.exists(): sys.exit(f"missing {SCRIPT}")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"reading {SCRIPT.name}, saving to {TAKES}/  ->  http://localhost:{PORT}   (Ctrl-C to stop)")
    threading.Timer(0.6, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nstopped;", len(takes_state()), "takes on disk")
