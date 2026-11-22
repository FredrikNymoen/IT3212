"""Serve local Part 2 results and render the recorded ALE action traces."""

import argparse
import csv
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import re
import tempfile
import threading
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

import ale_py
import gymnasium as gym
import imageio_ffmpeg
from PIL import Image


HERE = Path(__file__).resolve().parent
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
NUMBER = re.compile(r"-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CACHE_FORMAT = "rgb-decisions-v1"

gym.register_envs(ale_py)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_value(value):
    if value in (None, "", "nan"):
        return None
    if NUMBER.fullmatch(value):
        return float(value) if set(value).intersection(".eE") else int(value)
    return value


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{key: csv_value(value) for key, value in row.items() if key}
                for row in csv.DictReader(stream)]


class DashboardError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class Results:
    def __init__(self, root):
        self.root = root.resolve()

    def experiments(self):
        if not self.root.is_dir():
            return []
        rows = []
        for path in sorted(self.root.iterdir(), reverse=True):
            if not TOKEN.fullmatch(path.name) or not (path / "experiment.json").is_file():
                continue
            try:
                manifest = read_json(path / "experiment.json")
            except (OSError, ValueError):
                continue
            agents = read_csv(path / "per_agent.csv")
            rows.append(dict(id=path.name, ready=bool(agents), agents=len(agents),
                             environment=manifest.get("env_id"), goal=manifest.get("success_points"),
                             label=manifest.get("label")))
        return rows

    def folder(self, experiment):
        if not TOKEN.fullmatch(experiment):
            raise DashboardError("Ugyldig kjørings-ID.", 404)
        folder = (self.root / experiment).resolve()
        if folder.parent != self.root or not (folder / "experiment.json").is_file():
            raise DashboardError("Fant ikke kjøringen.", 404)
        return folder

    def overview(self, folder):
        agents = read_csv(folder / "per_agent.csv")
        runs = {}
        for run_id in {row["run_id"] for row in agents}:
            path = folder / run_id / "run.json"
            if path.is_file():
                runs[run_id] = read_json(path)
        return dict(manifest=read_json(folder / "experiment.json"), agents=agents, runs=runs,
                    summary=read_csv(folder / "summary.csv"), curves=read_csv(folder / "learning_curves.csv"))

    def agent(self, folder, eval_id):
        for agent in read_csv(folder / "per_agent.csv"):
            if agent["eval_id"] == eval_id:
                return agent
        raise DashboardError("Fant ikke evalueringen i denne kjøringen.", 404)

    def episode_files(self, folder, agent):
        run = (folder / agent["run_id"]).resolve()
        setting = str(agent["setting"])
        if run.parent != folder or not TOKEN.fullmatch(setting):
            raise DashboardError("Ugyldig plassering av episodeloggen.")
        return run / f"evaluation_{setting}.csv", run / f"actions_{setting}.jsonl.gz"

    def replay(self, folder, eval_id, seed):
        agent = self.agent(folder, eval_id)
        csv_path, trace_path = self.episode_files(folder, agent)
        episode = next((row for row in read_csv(csv_path) if row["eval_seed"] == seed), None)
        if episode is None:
            raise DashboardError("Fant ikke episodefrøet i denne evalueringen.", 404)
        if not trace_path.is_file():
            raise DashboardError("Kjøringen mangler et lagret handlingsspor for denne episoden.", 404)
        with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                trace = json.loads(line)
                if trace["eval_seed"] == seed:
                    break
            else:
                raise DashboardError("Fant ikke handlingene for denne episoden.", 404)
        manifest = read_json(folder / "experiment.json")
        return Replay(manifest, agent["setting"], episode, trace["actions"])


class Replay:
    def __init__(self, manifest, setting, episode, actions):
        self.env_id = manifest["env_id"]
        if self.env_id != "ALE/MsPacman-v5":
            raise DashboardError("Avspilleren støtter ALE/MsPacman-v5.")
        self.setting = manifest["settings"][setting]
        self.frame_skip = manifest["frame_skip"]
        self.fps = 60 / self.frame_skip
        self.seed = episode["eval_seed"]
        self.record = episode
        self.actions = actions
        self.goal = manifest["success_points"]
        self.limit = manifest["max_episode_steps"]
        data = [CACHE_FORMAT, self.env_id, self.setting, self.frame_skip, self.goal, self.limit, episode,
                actions, ale_py.__version__]
        self.key = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:32]
        if len(actions) != episode["length"] or not 0 < len(actions) <= self.limit:
            raise DashboardError("Lengden på handlingssporet stemmer ikke med evalueringen.", 422)

    def frames(self):
        with gym.make(self.env_id, obs_type="grayscale", frameskip=self.frame_skip,
                      repeat_action_probability=self.setting["sticky"], full_action_space=False) as env:
            ale = env.unwrapped.ale
            actions = env.unwrapped.get_action_meanings()
            env.reset(seed=self.seed)
            if self.setting["maze"]:
                ale.setRAM(0, self.setting["maze"])
            _, points, game_over, timeout, info = env.step(actions.index("NOOP"))
            if points or game_over or timeout or ale.getRAM()[0] != self.setting["maze"]:
                raise DashboardError("Emulatoren kunne ikke gjenskape starttilstanden.", 422)
            score = 0.0
            yield ale.getScreenRGB(), dict(score=0, lives=info["lives"], action=None)
            for step, action in enumerate(self.actions, 1):
                if not env.action_space.contains(action):
                    raise DashboardError("Handlingssporet inneholder en ugyldig handling.", 422)
                _, points, game_over, timeout, info = env.step(action)
                score += points
                cleared = score >= self.goal
                maze_exit = ale.getRAM()[0] != self.setting["maze"]
                result = ("success" if cleared else "game_over" if game_over else
                          "maze_exit" if maze_exit else "timeout" if timeout or step == self.limit else None)
                if result is not None and step != len(self.actions):
                    raise DashboardError("Episoden endte tidligere enn handlingssporet.", 422)
                yield ale.getScreenRGB(), dict(score=score, lives=info["lives"], action=actions[action])
            reward = (float(result == "success") + self.setting.get("score_bonus", 0.0) * score / self.goal
                      - len(self.actions) * self.setting["step_penalty"])
            if (result != self.record["outcome"] or score != self.record["raw_score"] or
                    not math.isclose(reward, self.record["episode_return"], abs_tol=1e-8)):
                raise DashboardError("Avspillingen stemte ikke med resultatloggen. Sjekk ALE-versjonen.", 422)


class MediaCache:
    def __init__(self, path):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def poster(self, replay):
        path = self.path / f"{replay.key}.png"
        if not path.is_file():
            with self.lock:
                if not path.is_file():
                    target = min(90, len(replay.actions) // 3)
                    generator = replay.frames()
                    try:
                        for step, (frame, _) in enumerate(generator):
                            if step == target:
                                buffer = io.BytesIO()
                                Image.fromarray(frame).save(buffer, format="PNG")
                                self.write(path, buffer.getvalue())
                                break
                    finally:
                        generator.close()
        return path

    def video(self, replay):
        clip = self.path / f"{replay.key}.mp4"
        metadata = self.path / f"{replay.key}.json"
        if clip.is_file() and metadata.is_file():
            return read_json(metadata)
        with self.lock:
            if clip.is_file() and metadata.is_file():
                return read_json(metadata)
            with tempfile.TemporaryDirectory(dir=self.path) as stage:
                temp_video = Path(stage) / clip.name
                timeline = []
                generator = replay.frames()
                first_frame, initial = next(generator)
                height, width = first_frame.shape[:2]
                writer = imageio_ffmpeg.write_frames(
                    str(temp_video), (width, height), fps=replay.fps, codec="libx264",
                    quality=8, macro_block_size=2,
                    output_params=["-preset", "ultrafast", "-movflags", "+faststart"],
                )
                writer.send(None)
                try:
                    writer.send(first_frame.tobytes())
                    timeline.append(initial)
                    for frame, detail in generator:
                        writer.send(frame.tobytes())
                        timeline.append(detail)
                finally:
                    generator.close()
                    writer.close()
                result = dict(video=f"/media/{clip.name}", fps=replay.fps, timeline=timeline)
                temp_video.replace(clip)
                self.write(metadata, json.dumps(result, separators=(",", ":")).encode())
                return result

    def write(self, path, data):
        with tempfile.NamedTemporaryFile(dir=self.path, delete=False) as stream:
            name = Path(stream.name)
            stream.write(data)
        try:
            name.replace(path)
        finally:
            name.unlink(missing_ok=True)


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def send_video(self, path):
        size = path.stat().st_size
        first, last = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if match and match[1]:
                first = int(match[1])
                last = min(last, int(match[2])) if match[2] else last
            elif match and match[2]:
                first = max(0, size - int(match[2]))
            else:
                first = size
            if not 0 <= first <= last < size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Cache-Control", "private, max-age=3600")
        if range_header:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(first)
            remaining = last - first + 1
            while remaining:
                chunk = stream.read(min(remaining, 64 * 1024))
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def dispatch(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        if path in STATIC:
            name, content_type = STATIC[path]
            return self.send_bytes((HERE / "web" / name).read_bytes(), content_type)
        if path == "/api/experiments":
            return self.send_json(self.server.results.experiments())
        if re.fullmatch(r"/media/[a-f0-9]{32}\.mp4", path):
            clip = self.server.media.path / Path(path).name
            if clip.is_file():
                return self.send_video(clip)
        match = re.fullmatch(r"/api/experiments/([^/]+)(?:/(episodes|poster|replay|summary\.csv))?", path)
        if not match:
            raise DashboardError("Fant ikke innholdet.", 404)
        experiment, operation = match.groups()
        results = self.server.results
        folder = results.folder(experiment)
        if operation is None:
            return self.send_json(results.overview(folder))
        if operation == "summary.csv":
            summary = folder / operation
            if not summary.is_file():
                raise DashboardError("Kjøringen har ingen resultattabell ennå.", 404)
            return self.send_bytes(summary.read_bytes(), "text/csv; charset=utf-8")
        query = parse_qs(url.query)
        eval_id = query.get("agent", [""])[0]
        if operation == "episodes":
            agent = results.agent(folder, eval_id)
            episodes, trace = results.episode_files(folder, agent)
            return self.send_json(dict(episodes=read_csv(episodes), has_trace=trace.is_file()))
        try:
            seed = int(query.get("seed", [""])[0])
        except ValueError as exc:
            raise DashboardError("Velg et gyldig episodefrø.") from exc
        replay = results.replay(folder, eval_id, seed)
        if operation == "poster":
            return self.send_bytes(self.server.media.poster(replay).read_bytes(), "image/png")
        return self.send_json(self.server.media.video(replay))

    def do_GET(self):
        try:
            self.dispatch()
        except DashboardError as exc:
            self.send_json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except (KeyError, ValueError, OSError) as exc:
            self.log_error("%s", exc)
            self.send_json({"error": "Dataene kunne ikke leses. Sjekk at kjøringen er ferdig."}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE.parent / "part2_results")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the local dashboard in the default browser.")
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.results = Results(args.results)
        server.media = MediaCache(Path(tempfile.gettempdir()) / "mspacman-dashboard-media")
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"MsPacMan Lab · {url}\nResultater · {server.results.root}", flush=True)
        if args.open:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
