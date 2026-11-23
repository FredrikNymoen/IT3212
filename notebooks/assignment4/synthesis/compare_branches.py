"""Recheck the reviewed Git sources without executing other branches' code."""
import csv
import hashlib
import io
import json
import statistics
import subprocess

from part2_evidence import ROOT, HERE, require


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT)


def notebook_code(data):
    nb = json.loads(data)
    return "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")


def main():
    sources = json.loads((HERE / "branch_sources.json").read_text(encoding="utf-8"))["sources"]
    verified = 0
    for branch in sources.values():
        for source in branch["files"]:
            blob = git("show", branch["commit"] + ":" + source["path"])
            require(hashlib.sha256(blob).hexdigest() == source["sha256"], "Reviewed source changed")
            verified += 1
    shared = [git("show", sources[name]["commit"] + ":" + path) for name, path in [
        ("johan", "notebooks/assignment4/RL-policy-based(1).ipynb"),
        ("elling", "RL-policy-based.ipynb")]]
    require(notebook_code(shared[0]) == notebook_code(shared[1]), "Shared policy code no longer identical")
    johan = json.loads(git("show", sources["johan"]["commit"] + ":notebooks/assignment4/RL-API-final.ipynb"))
    part2_cells = johan["cells"][26:]
    require(all(not cell.get("outputs") for cell in part2_cells if cell["cell_type"] == "code")
            and "RUN_PART2_SMOKE = False" in notebook_code(json.dumps(johan))
            and "RUN_PART2_FULL = False" in notebook_code(json.dumps(johan)),
            "Part 2 launch/output status differs from review")
    revision = sources["mats"]["commit"]
    paths = git("ls-tree", "-rz", "--name-only", revision).decode("utf-8").split("\0")
    outcomes = {}
    for study in ["assignment4_v8", "assignment4_v8_500k_parallel"]:
        groups = {}
        prefix = "øv4/results/" + study + "/"
        files = [path for path in paths if path.startswith(prefix)
                 and "/final_" in path and path.endswith("_episodes.csv")]
        require(len(files) == 40, "Expected all 40 Mats primary cells")
        for path in files:
            rows = list(csv.DictReader(io.StringIO(git("show", revision + ":" + path).decode("utf-8"))))
            require(len(rows) == 200, "Expected 200 Mats episodes per cell")
            run = path.split("/")[-2]
            algorithm = run.split("_")[1]
            condition = path.split("/")[-1].split("_")[1]
            require(algorithm in ("Q", "PPO") and condition in "ABCD", "Unexpected comparison arm")
            rate = sum(row["outcome"] == "success" for row in rows) / len(rows)
            groups.setdefault(algorithm + "/" + condition, []).append(rate)
        require(all(len(values) == 5 for values in groups.values()), "Expected five training replications")
        outcomes[study] = {key: dict(mean_success_percent=100 * statistics.mean(values),
                                    training_seeds=5, episodes_per_model=200)
                           for key, values in sorted(groups.items())}
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    mazes = {}
    for mode in range(4):
        env = gym.make("ALE/MsPacman-v5", obs_type="ram", frameskip=4, mode=mode)
        try:
            env.reset(seed=42)
            mazes[str(mode)] = int(env.unwrapped.ale.getRAM()[0])
        finally:
            env.close()
    require(set(mazes.values()) == {0}, "ALE mode/reset maze relationship differs from reviewed version")
    frontier = json.loads(git("show", revision + ":øv4/results/assignment4_ppo_hour_frontier_B/comparison.json"))
    averaged = json.loads(git("show", revision + ":øv4/results/assignment4_late_averaged_confirmation/comparison.json"))
    cold = [r for r in frontier["runs"] if r["name"] == "cold_30k"]
    late = averaged["runs"]
    require(len(cold) == len(late) == 3, "Later studies have three trained seeds")
    require(all(r["episodes"] == 300 and r["temperature"] == .2 for r in cold), "Frontier policy changed")
    require(all(r["episodes"] == 400 and r["layout"] == 1 for r in late), "Late transfer policy changed")
    cold_mean = 100 * statistics.mean(r["success_rate"] for r in cold)
    late_mean = 100 * statistics.mean(r["success_rate"] for r in late)
    require(round(cold_mean, 1) == 98.3 and round(late_mean, 1) == 48.8, "Later-study summary changed")
    result = dict(branches=4, reviewed_source_blobs=verified, other_branch_code_executed=False,
                  shared_policy_code_identical=True, johan_Part2_launch_flags_off_and_no_result_outputs=True,
                  native_reset_maze_by_ALE_mode=mazes, mats_primary_outcomes=outcomes,
                  mats_later_unmatched=dict(cold_30k_B_success_percent=cold_mean,
                      averaged_transfer_success_percent=late_mean, training_seeds=3,
                      primary_report_replacement=False),
                  source_commits={name: value["commit"] for name, value in sources.items()},
                  scope="Protocol review only. Outcomes from incompatible studies are not pooled into the common report.")
    output = ROOT / "results/assignment4/synthesis/branch_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
