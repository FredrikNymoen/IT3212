"""Verify the synthesis against immutable original evidence; never train."""
import contextlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

from part2_evidence import (ROOT, HERE, BUNDLE, ALGORITHMS, TRAIN_SEEDS,
                            build_bundle, check_bundle, dashboard, load_bundle,
                            local_path, require, sha, read_csv)
from replay import replay


def rejects(call, explanation):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("Expected rejection: " + explanation)


def main():
    output = ROOT / "results/assignment4/synthesis" / ("validation_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    original = BUNDLE.read_bytes()
    checks = check_bundle(full=True)
    bundle = load_bundle()
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temp = Path(temporary)
        rebuilt = build_bundle(temp / "evidence.json")
        require(rebuilt == bundle, "Tracked evidence differs from current raw archive")
        tampered = json.loads(original)
        tampered["agents"][0]["success_rate"] = .12345
        (temp / "tampered.json").write_text(json.dumps(tampered), encoding="utf-8")
        rejects(lambda: load_bundle(temp / "tampered.json"), "tampered summary")
        require(local_path(r"X:\old\IT3212\results\assignment4\transfer_boundary_v1")
                == ROOT / "results/assignment4/transfer_boundary_v1", "Archived path not portable")
        rejects(lambda: local_path(r"..\outside"), "parent traversal")
        rejects(lambda: local_path(r"X:\unrelated\file.pkl"), "unrelated absolute path")
        dashboard(bundle, temp / "overview.html")
        require("/* EVIDENCE_DATA */null" not in (temp / "overview.html").read_text(encoding="utf-8"),
                "Dashboard data not embedded")
        node = shutil.which("node")
        require(node is not None, "Node is required for the HTML JavaScript syntax check")
        for name in ["dashboard.html", "replay.html"]:
            content = (HERE / name).read_text(encoding="utf-8")
            script = re.search(r"<script>(.*?)</script>", content, re.S).group(1)
            path = temp / (name + ".js")
            path.write_text(script, encoding="utf-8")
            subprocess.run([node, "--check", str(path)], check=True, capture_output=True, text=True)
        # Execute every code cell using the current Python process and notebook cwd.
        # Rendering objects are created normally, but display output is captured.
        import IPython.display
        notebook = ROOT / "notebooks/assignment4/02_part2_synthesis.ipynb"
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        namespace = {"__name__": "__synthesis_notebook__"}
        saved_display = IPython.display.display
        displayed = []
        IPython.display.display = lambda *objects, **kwargs: displayed.extend(type(obj).__name__ for obj in objects)
        count = 0
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in nb["cells"]:
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), str(notebook), "exec"), namespace)
                        count += 1
        finally:
            IPython.display.display = saved_display
        require(count == 5 and len(displayed) == 4, "Notebook code cells/displays incomplete")
    cases = []
    # Fixed identities, not chosen by final outcomes or highest test score.
    for condition in "ABCD":
        for algorithm in ALGORITHMS:
            seed = 1250000 if condition == "D" else 1150000
            cases.append(replay(condition, algorithm, TRAIN_SEEDS[0], seed,
                                output / f"{condition}_{algorithm}_17.html", stride=12))
    # Include every censored episode, because the map boundary is a distinct contract.
    for case in bundle["agents"]:
        if not case["censored"]:
            continue
        for row in read_csv(local_path(case["evaluation_csv"])):
            if int(row.get("censored", 0)):
                cases.append(replay(case["condition"], case["algorithm"], case["training_seed"],
                                    int(row["evaluation_seed"]), output / "censored_Q_D.html", stride=12))
    require(len(cases) == 9, "Expected eight fixed cases and the sole censored case")
    require(BUNDLE.read_bytes() == original, "Verification changed the tracked evidence snapshot")
    result = dict(checks, raw_rebuild_matches_tracked_snapshot=True, unsafe_paths_and_tampered_summary_rejected=True,
                  notebook_code_cells_executed=count, notebook_display_objects=displayed,
                  html_javascript_syntax_checked=True, native_replayed_episodes=len(cases),
                  native_decisions=sum(r["verified_native_decisions"] for r in cases), cases=cases,
                  new_training=False, new_evaluation=False, output=str(output.relative_to(ROOT)))
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
