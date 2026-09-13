#!/usr/bin/env python3
"""RunPod REST API v2 helper.

Replaces the GraphQL path used by whisper-train/pod_*.py. The reason is not the
deprecation date (GraphQL retires early 2027, REST v1 on 2026-11-15) but that
GraphQL lied to us: on 2026-08-19 `podTerminate` returned
`{"podTerminate": null}` and `myself { pods }` returned `[]` while the pod kept
running and billing $0.34/hr for ten more hours, $3.59 instead of $0.07. The
on-pod watchdog missed it too, so nothing caught it until `/v2/billing/pods`
existed to ask the only question that cannot be faked: is this still costing
money?

Hence terminate_verified(): terminate, then confirm against billing.
"""
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://api.runpod.io"
KEYFILE = pathlib.Path.home() / ".config/runpod/env"
SSH_KEY = pathlib.Path(os.environ.get("POD_SSH_KEY", pathlib.Path.home() / ".ssh/id_ed25519"))


def _key() -> str:
    for line in KEYFILE.read_text().splitlines():
        if line.startswith("RUNPOD_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no RUNPOD_API_KEY")


def api(method: str, path: str, body=None, params=None):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
        "User-Agent": "curl/8.6.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw.decode(errors="replace")[:400]


def create_pod(name, gpu_id, image, disk_gb=120, cloud="SECURE", env=None):
    """Returns the pod dict. Raises with the API's own message on failure."""
    pub = SSH_KEY.with_suffix(".pub").read_text().strip()
    body = {
        "name": name,
        "image": image,
        "gpu": {"id": gpu_id, "count": 1},
        "disk": disk_gb,
        "cloud": cloud,
        "startSsh": True,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pub, **(env or {})},
    }
    st, d = api("POST", "/v2/pods", body)
    if st not in (200, 201):
        raise RuntimeError(f"create failed {st}: {json.dumps(d)[:400]}")
    return d.get("pod", d)


def wait_ssh(pod_id, timeout_s=900):
    """Poll until the pod exposes a public port 22. Returns (ip, port)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st, d = api("GET", f"/v2/pods/{pod_id}")
        pod = (d or {}).get("pod", d) or {}
        for p in (pod.get("portMappings") or []) if isinstance(pod.get("portMappings"), list) else []:
            if str(p.get("privatePort")) == "22" and p.get("publicPort"):
                return p.get("ip"), p["publicPort"]
        # 2026-09 shape: runtime.ports = [{ip, private, public, type}]
        rt = pod.get("runtime") or {}
        for p in (rt.get("ports") or []) if isinstance(rt, dict) else []:
            if str(p.get("private")) == "22" and p.get("public"):
                return p.get("ip"), p["public"]
        # shape varies by response version; fall back to a scan
        blob = json.dumps(pod)
        if '"22"' in blob or "22/tcp" in blob:
            ports = pod.get("ports") or pod.get("publicPorts") or {}
            if isinstance(ports, dict) and ports.get("22"):
                e = ports["22"]
                return e.get("ip"), e.get("publicPort")
        time.sleep(10)
    raise RuntimeError("ssh never appeared")


def billed_recently(pod_id, since_iso, until_iso):
    """Total billed for this pod in the window. The honest liveness check."""
    st, d = api("GET", "/v2/billing/pods",
                params={"podId": pod_id, "bucketSize": "hour",
                        "startTime": since_iso, "endTime": until_iso})
    if st != 200 or not isinstance(d, dict):
        return None
    return (d.get("metadata", {}).get("totals", {}) or {}).get("totalAmount")


def terminate_verified(pod_id, since_iso, until_iso, settle_s=90):
    """Terminate, then prove it stopped costing money.

    A terminate call that returns 200 is not proof; that is exactly what
    happened on 2026-08-19. This waits, then checks the pod is gone AND that
    billing has stopped advancing.
    """
    st, d = api("DELETE", f"/v2/pods/{pod_id}")
    out = {"terminate_status": st, "terminate_body": d}
    before = billed_recently(pod_id, since_iso, until_iso)
    time.sleep(settle_s)
    st2, _ = api("GET", f"/v2/pods/{pod_id}")
    after = billed_recently(pod_id, since_iso, until_iso)
    # Billing lags by up to an hour, so "before == after == 0" proves nothing:
    # treating that as confirmation would rebuild the exact false confidence
    # this function exists to prevent. Only a NON-ZERO reading that stops moving
    # is evidence. A 404 on the pod is the strong signal; billing corroborates
    # it later, which is why check_still_billing() exists for a follow-up pass.
    billing_meaningful = bool(before) and bool(after)
    out.update(pod_lookup_status=st2, billed_before=before, billed_after=after,
               gone=(st2 == 404),
               billing_meaningful=billing_meaningful,
               billing_stopped=(billing_meaningful and abs(after - before) < 0.01))
    out["verified_dead"] = bool(out["gone"])
    out["note"] = ("pod is gone; billing had not reported yet, so re-check with "
                   "check_still_billing() in an hour"
                   if not billing_meaningful else "pod gone and billing flat")
    return out


def check_still_billing(pod_id, since_iso, until_iso):
    """Follow-up pass once billing has settled. Returns (total, is_growing)."""
    a = billed_recently(pod_id, since_iso, until_iso)
    time.sleep(120)
    b = billed_recently(pod_id, since_iso, until_iso)
    return b, (a is not None and b is not None and b - a > 0.01)


def ssh_cmd(ip, port):
    return ["ssh", "-i", str(SSH_KEY), "-p", str(port),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20", f"root@{ip}"]


def scp_cmd(ip, port):
    return ["scp", "-i", str(SSH_KEY), "-P", str(port),
            "-o", "StrictHostKeyChecking=accept-new"]


def run_ssh(ip, port, command, timeout=120):
    return subprocess.run(ssh_cmd(ip, port) + [command],
                          capture_output=True, text=True, timeout=timeout)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "pods":
        print(json.dumps(api("GET", "/v2/pods")[1], indent=1))
    elif len(sys.argv) > 2 and sys.argv[1] == "kill":
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        day = time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime())
        print(json.dumps(terminate_verified(sys.argv[2], day, now), indent=1))
    else:
        print(__doc__)
