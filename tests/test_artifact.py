import subprocess, json, sys

def test_runs_quickly_and_correctly():
    cmd = [sys.executable, "artifact.py", "--runs", "1", "--n", "2000", "--dim", "64", "--topk", "5", "--csv", "bench.csv"]
    out = subprocess.check_output(cmd, timeout=60).decode()
    j = json.loads(out)
    assert j["correct"] == 1
    assert 0.0 <= j["metric"] <= 1.0
    assert j["runtime_s"] < 5.0
