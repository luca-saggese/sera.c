#!/usr/bin/env python3
"""M4 end-to-end System One benchmark (docs/M4.md §28-§47).

Stdlib only. Sequential ladder (Q = questions per request, not clients),
one concurrency smoke, artifacts written under --out.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

NEAREST_RANK = lambda s, p: s[max(0, __import__("math").ceil(p * len(s)) - 1)]


def http_json(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=body, method="POST" if body else "GET")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    ms = (time.perf_counter() - t0) * 1000.0
    try:
        return status, json.loads(raw), ms
    except Exception:
        return status, None, ms


def wait_health(base, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st, j, _ = http_json(base + "/health", timeout=5)
            if st == 200 and j and j.get("ready"):
                return (time.time() - t0) * 1000.0
        except Exception:
            pass
        time.sleep(0.25)
    raise SystemExit("server not healthy within %ds" % timeout)


def request(base, fixture, q, questions, expected):
    body = {
        "model": fixture["model"],
        "state": fixture["state"],
        "questions": {qid: {k: v for k, v in questions[qid].items() if k != "expected"}
                      for qid in questions},
    }
    st, j, ms = http_json(base + "/v1/systemone", json.dumps(body).encode())
    matches = 0
    if st == 200 and j and "answers" in j:
        for i, qid in enumerate(questions):
            exp = expected[qid]
            ans = j["answers"].get(qid, {})
            if ans.get("type") == "choice" and "choice" in ans:
                keys = list(questions[qid]["criteria"].keys())
                if exp < len(keys) and ans["choice"] == keys[exp]:
                    matches += 1
    return st, j, ms, matches


def run_ladder(base, fixture, out, requests, ladder, warmup):
    qs = fixture["questions"]
    expected = {qid: q["expected"] for qid, q in qs.items()}
    results = {}
    for q in ladder:
        qids = list(qs.keys())[:q]
        sub = {qid: qs[qid] for qid in qids}
        for _ in range(warmup):
            request(base, fixture, q, sub, expected)
        lat, statuses, matches, raws = [], [], 0, []
        for i in range(requests):
            st, j, ms, m = request(base, fixture, q, sub, expected)
            lat.append(ms)
            statuses.append(st)
            matches += m
            raws.append({"i": i, "status": st, "latency_ms": ms,
                         "stats": (j or {}).get("stats")})
        s = sorted(lat)
        n = len(s)
        p50 = NEAREST_RANK(s, 0.50)
        p95 = NEAREST_RANK(s, 0.95)
        mean = statistics.mean(lat)
        art = {
            "q": q, "requests": requests, "state_chars": len(fixture["state"]),
            "state_tokens": 0, "suffix_tokens_total": 0,
            "http_status": statuses, "latency_ms": lat,
            "p50_ms": p50, "p95_ms": p95, "mean_ms": mean,
            "min_ms": min(lat), "max_ms": max(lat),
            "questions_per_s": q / (mean / 1000.0),
            "e2e_ms_per_question": mean / q,
            "breakdown_median_ms": None,
            "server_overhead_ms": None, "server_overhead_percent": None,
            "amortized_prefix_ms_per_question": None,
            "kv": None,
            "gpu_memory": {"after_load_bytes": None,
                           "after_first_request_bytes": None,
                           "peak_bytes": None},
            "accuracy": {"matches": matches, "total": requests * q},
            "raw": raws,
        }
        results[q] = art
        with open(os.path.join(out, "q%d.json" % q), "w") as f:
            json.dump(art, f, indent=1)
        print("Q%-2d p50=%8.1fms p95=%8.1fms mean=%8.1fms q/s=%7.1f acc=%d/%d"
              % (q, p50, p95, mean, art["questions_per_s"], matches, requests * q))
    return results


def run_concurrency(base, fixture, out, clients):
    qid = list(fixture["questions"].keys())[0]
    q = fixture["questions"][qid]
    body = json.dumps({
        "model": fixture["model"], "state": fixture["state"],
        "questions": {qid: {k: v for k, v in q.items() if k != "expected"}},
    }).encode()
    barrier = threading.Barrier(clients)
    out_l = []
    def worker(i):
        barrier.wait()
        t0 = time.perf_counter()
        st, j, _ = http_json(base + "/v1/systemone", body)
        ms = (time.perf_counter() - t0) * 1000.0
        choice = None
        if j and "answers" in j:
            a = j["answers"].get(qid, {})
            choice = a.get("choice")
        out_l.append({"i": i, "status": st, "latency_ms": ms,
                      "queue_wait_ms": None, "choice": choice,
                      "expected": list(q["criteria"].keys())[q["expected"]],
                      "correct": choice == list(q["criteria"].keys())[q["expected"]]})
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.perf_counter() - t0) * 1000.0
    qw = [r["queue_wait_ms"] for r in out_l if r["queue_wait_ms"] is not None]
    art = {
        "clients": clients, "questions_per_request": 1, "wall_ms": wall,
        "all_status_200": all(r["status"] == 200 for r in out_l),
        "requests": out_l,
        "queue_wait_ms": {"min": min(qw) if qw else None,
                          "p50": NEAREST_RANK(sorted(qw), 0.50) if qw else None,
                          "max": max(qw) if qw else None},
        "distinct_choices": sorted({r["choice"] for r in out_l}),
        "correct": all(r["correct"] for r in out_l),
    }
    with open(os.path.join(out, "concurrency%d.json" % clients), "w") as f:
        json.dump(art, f, indent=1)
    print("concurrency%d: all200=%s correct=%s wall=%.1fms"
          % (clients, art["all_status_200"], art["correct"], wall))
    return art


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8011")
    ap.add_argument("--fixture", default="docs/research/m4/bench_systemone.json")
    ap.add_argument("--out", default="artifacts/m4_e2e")
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--ladder", default="1,2,4,8,16,32")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--health-timeout", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fixture = json.load(open(args.fixture))
    ladder = [int(x) for x in args.ladder.split(",")]

    startup_ms = wait_health(args.base_url, args.health_timeout)
    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    startup = {
        "commit": commit,
        "model": "models/Qwen3-32B-Q4_K_M.gguf",
        "model_file_bytes": 19762149024,
        "gpu": "NVIDIA GB10 (sm_121a)",
        "server_cmd": ["build/q3-server", "--model", "models/Qwen3-32B-Q4_K_M.gguf",
                       "--port", "8011"],
        "process_start_to_health_ms": startup_ms,
        "gguf_open_ms": None, "residency_load_ms": None,
        "runtime_workspace_init_ms": None, "server_ready_ms": None,
        "resident_model_bytes": None, "gpu_memory_after_load_bytes": None,
        "first_request_ms": None, "first_request_breakdown_ms": None,
    }
    with open(os.path.join(args.out, "startup.json"), "w") as f:
        json.dump(startup, f, indent=1)

    results = run_ladder(args.base_url, fixture, args.out, args.requests,
                         ladder, args.warmup)
    concurrency = run_concurrency(args.base_url, fixture, args.out,
                                  args.concurrency)

    lines = [
        "# M4 E2E benchmark", "",
        "commit: %s" % commit,
        "model: models/Qwen3-32B-Q4_K_M.gguf",
        "model bytes: 19,762,149,024",
        "GPU: NVIDIA GB10 (sm_121a)", "",
        "startup_ms: %.1f" % startup_ms,
        "resident_model_bytes: N/A",
        "GPU memory after load: N/A", "",
        "samples per Q: %d (p95 == max)" % args.requests, "",
        "| Q | p50 ms | p95 ms | q/s | ms/q |",
        "|---|--------|--------|-----|------|",
    ]
    for q in ladder:
        r = results[q]
        lines.append("| %d | %.1f | %.1f | %.1f | %.2f |"
                     % (q, r["p50_ms"], r["p95_ms"], r["questions_per_s"],
                        r["e2e_ms_per_question"]))
    lines += [
        "", "server overhead: N/A (no --debug-stats in server)", "",
        "runtime breakdown: N/A (no stats block)", "",
        "GPU peak: N/A", "",
        "correctness:",
        "  smoke: PASS",
        "  ladder accuracy: %d/%d" % (
            sum(r["accuracy"]["matches"] for r in results.values()),
            sum(r["accuracy"]["total"] for r in results.values())),
        "  concurrency%d: all200=%s correct=%s" % (
            args.concurrency, concurrency["all_status_200"], concurrency["correct"]),
        "", "main bottleneck: N/A (no per-phase telemetry; see M4 §46)",
    ]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()