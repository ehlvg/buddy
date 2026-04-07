import argparse
import contextlib
import json
import os
import queue as _queue
import shlex
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard as kb

# ── constants ──────────────────────────────────────────────────────────────────

SAMPLERATE        = 44100
CHANNELS          = 1
DTYPE             = "int16"
BLOCKSIZE         = 1024
MIN_DURATION      = 0.3
SILENCE_THRESHOLD = 300

CONFIG_PATH = Path.home() / ".config" / "buddy" / "config.json"
FORMATS     = ("wav", "flac", "ogg", "mp3")
SPINNER     = ("|", "/", "-", "\\")

# ── colors ─────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
PINK   = "\033[35m"
CYAN   = "\033[36m"
GRAY   = "\033[90m"

METER_WIDTH = 20

# ── config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}

def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))

# ── audio utils ────────────────────────────────────────────────────────────────

def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

def write_audio(path: Path, audio: np.ndarray, fmt: str):
    if fmt == "mp3":
        try:
            import lameenc
        except ImportError:
            raise RuntimeError("mp3 requires lameenc — pip install lameenc")
        enc = lameenc.Encoder()
        enc.set_bit_rate(192)
        enc.set_in_sample_rate(SAMPLERATE)
        enc.set_channels(CHANNELS)
        enc.set_quality(2)
        path.write_bytes(enc.encode(audio.tobytes()) + enc.flush())
    elif fmt == "ogg":
        sf.write(str(path), audio, SAMPLERATE, format="OGG", subtype="VORBIS")
    elif fmt == "flac":
        sf.write(str(path), audio, SAMPLERATE, format="FLAC")
    else:
        sf.write(str(path), audio, SAMPLERATE)

def get_recordings(out_dir: Path) -> list:
    files = []
    for fmt in FORMATS:
        files.extend(out_dir.glob(f"*.{fmt}"))
    seen: dict = {}
    for f in files:
        r = str(f.resolve())
        if r not in seen:
            seen[r] = f
    result = list(seen.values())
    try:
        result.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        pass
    return result

# ── caches (single-writer-per-entry, GIL-safe in CPython) ─────────────────────

_dur_cache: dict = {}

def audio_duration(path: Path) -> Optional[float]:
    key = str(path)
    if key not in _dur_cache:
        try:
            _dur_cache[key] = sf.info(str(path)).duration
        except Exception:
            _dur_cache[key] = None
    return _dur_cache[key]

_tx_cache: dict = {}  # str(path) -> str|None

def read_transcript(path: Path) -> Optional[str]:
    key = str(path)
    if key in _tx_cache:
        return _tx_cache[key]
    txt = path.with_suffix(".txt")
    if txt.exists():
        try:
            text = txt.read_text(encoding="utf-8").strip()
            _tx_cache[key] = text or None
            return _tx_cache[key]
        except Exception:
            pass
    return None

def bust_transcript(path: Path):
    _tx_cache.pop(str(path), None)

# ── transcription worker ───────────────────────────────────────────────────────

class TranscriptionWorker:
    def __init__(self, model_size: str = "base"):
        self._q: _queue.Queue = _queue.Queue()
        self.current: Optional[Path] = None   # written by worker, read by render
        self._model_size = model_size
        self._model = None
        self._unavailable = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, path: Path):
        if not self._unavailable:
            self._q.put(path)

    @property
    def queue_size(self) -> int:
        return self._q.qsize()

    @property
    def pending(self) -> int:
        return self._q.qsize() + (1 if self.current else 0)

    def _load(self):
        if self._model:
            return self._model
        try:
            from faster_whisper import WhisperModel  # type: ignore
            m = WhisperModel(self._model_size, device="cpu", compute_type="int8")
            self._model = ("fw", m)
            return self._model
        except Exception:
            pass
        try:
            import whisper  # type: ignore
            m = whisper.load_model(self._model_size)
            self._model = ("ow", m)
            return self._model
        except Exception:
            pass
        self._unavailable = True
        return None

    def _transcribe(self, path: Path) -> Optional[str]:
        loaded = self._load()
        if not loaded:
            return None
        tag, model = loaded
        try:
            if tag == "fw":
                segs, _ = model.transcribe(str(path), beam_size=5)
                return " ".join(s.text.strip() for s in segs).strip() or None
            else:
                res = model.transcribe(str(path))
                return (res.get("text") or "").strip() or None
        except Exception:
            return None

    def _run(self):
        while True:
            path = self._q.get()
            self.current = path
            try:
                bust_transcript(path)
                text = self._transcribe(path)
                if text:
                    path.with_suffix(".txt").write_text(text, encoding="utf-8")
                bust_transcript(path)
            except Exception:
                pass
            finally:
                self.current = None
                self._q.task_done()

# ── audio player ───────────────────────────────────────────────────────────────

class AudioPlayer:
    def __init__(self):
        self.playing = False
        self.current: Optional[Path] = None
        self._t: Optional[threading.Thread] = None

    def play(self, path: Path):
        self.stop()
        self._t = threading.Thread(target=self._play, args=(path,), daemon=True)
        self._t.start()

    def _play(self, path: Path):
        self.playing = True
        self.current = path
        try:
            data, sr = sf.read(str(path))
            sd.play(data, sr)
            sd.wait()
        except Exception:
            pass
        finally:
            self.playing = False
            self.current = None

    def stop(self):
        if self.playing:
            sd.stop()
        if self._t and self._t.is_alive():
            self._t.join(timeout=1.0)
        self.playing = False
        self.current = None

# ── hooks ──────────────────────────────────────────────────────────────────────

def run_hook(template: Optional[str], path: Path):
    if not template:
        return
    # Replace {file} token-by-token to handle paths with spaces safely
    parts = [t.replace("{file}", str(path)) for t in shlex.split(template)]
    def _exec():
        try:
            subprocess.run(parts, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    threading.Thread(target=_exec, daemon=True).start()

# ── TUI helpers ────────────────────────────────────────────────────────────────

def clear():
    print("\033[2J\033[H", end="", flush=True)

def level_bar(value: float, max_val: float = 3000.0) -> str:
    filled = int(min(value / max_val, 1.0) * METER_WIDTH)
    empty  = METER_WIDTH - filled
    color  = RED if filled > METER_WIDTH * 0.75 else GREEN
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"

def fmt_path(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)

def spin(tick: int) -> str:
    return SPINNER[tick % len(SPINNER)]

def trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"

def header(subtitle: str = "voice recorder"):
    print(f"\n  {PINK}<(^o^<){RESET}  {BOLD}buddy{RESET}  {DIM}— {subtitle}{RESET}\n")

# ── draw functions (called only from render thread) ────────────────────────────

def draw_record(snap: dict):
    clear()
    header()
    print(f"  {DIM}mic:{RESET} {snap['device_name']}  "
          f"{DIM}· fmt:{RESET} {snap['fmt']}  "
          f"{DIM}· dir:{RESET} {fmt_path(snap['out_dir'])}\n")

    if snap["recording"]:
        print(f"  {RED}● {snap['rec_dur']:.1f}s{RESET}  "
              f"{level_bar(snap['current_rms'])}  "
              f"{DIM}release SPACE to save{RESET}\n")
    else:
        print(f"  {DIM}hold SPACE  ·  b browse  ·  Ctrl+C quit{RESET}\n")

    # transcription status
    cur_tx = snap["tx_current"]
    tx_q   = snap["tx_queue_size"]
    if cur_tx:
        q_str = f"  {DIM}({tx_q} queued){RESET}" if tx_q else ""
        print(f"  {CYAN}{spin(snap['tick'])}{RESET} "
              f"{DIM}transcribing{RESET} {cur_tx.name}{q_str}\n")
    elif tx_q:
        print(f"  {CYAN}{spin(snap['tick'])}{RESET} "
              f"{DIM}{tx_q} waiting to transcribe{RESET}\n")

    # status / warning line
    if snap["exit_warning"]:
        print(f"  {YELLOW}! transcriptions pending — Ctrl+C again to force quit{RESET}\n")
    elif not snap["recording"]:
        if snap["skipped"]:
            print(f"  {YELLOW}⚠ too quiet — nothing saved{RESET}\n")
        elif snap["last_saved"]:
            print(f"  {GREEN}✓ saved:{RESET} {snap['last_saved']}\n")

    if snap["saved"]:
        print(f"  {DIM}recent:{RESET}")
        for fname, dur in snap["saved"][-6:]:
            bar = "▪" * min(int(dur * 4), 40)
            print(f"  {DIM}│{RESET} {fname}  {DIM}{bar} {dur:.1f}s{RESET}")
            fpath = snap["out_dir"] / fname
            if cur_tx and cur_tx.name == fname:
                print(f"  {DIM}│  {CYAN}{spin(snap['tick'])} transcribing...{RESET}")
            else:
                t = read_transcript(fpath)
                if t:
                    print(f"  {DIM}│  {GRAY}[>] {trunc(t, 68)}{RESET}")
        print()


def draw_browse(snap: dict):
    clear()
    header("browse recordings")
    print(f"  {DIM}q back  ·  UP/DOWN select  ·  p play/stop{RESET}\n")
    print(f"  {DIM}dir:{RESET} {fmt_path(snap['out_dir'])}\n")

    files = snap["browse_files"]
    if not files:
        print(f"  {DIM}no recordings found{RESET}\n")
        return

    cur_tx = snap["tx_current"]
    for i, path in enumerate(files[:24]):
        sel  = (i == snap["browse_cursor"])
        pfx  = f"  {GREEN}>{RESET}" if sel else "   "
        dur  = audio_duration(path)
        bar  = f"  {DIM}{'▪' * min(int(dur * 4), 40)} {dur:.1f}s{RESET}" if dur else ""
        playing = snap["player_playing"] and snap["player_current"] == path
        pg   = f"  {CYAN}(playing){RESET}" if playing else ""
        name = f"{BOLD}{path.name}{RESET}" if sel else path.name

        print(f"{pfx} {name}{bar}{pg}")

        if cur_tx and cur_tx == path:
            print(f"     {CYAN}{spin(snap['tick'])} transcribing...{RESET}")
        else:
            t = read_transcript(path)
            if t:
                print(f"     {GRAY}[>] {trunc(t, 70)}{RESET}")

    print()


def render_loop(st: dict, lock: threading.Lock, should_exit: threading.Event):
    while not should_exit.is_set():
        with lock:
            snap = dict(st)
        if snap["mode"] == "record":
            draw_record(snap)
        else:
            draw_browse(snap)
        time.sleep(0.1)

# ── device picker ──────────────────────────────────────────────────────────────

def list_inputs():
    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]

def pick_device(cfg: dict, force: bool = False):
    inputs = list_inputs()
    if not inputs:
        print(f"  {RED}no input devices found{RESET}")
        sys.exit(1)

    if not force and (saved_name := cfg.get("device")):
        for i, d in inputs:
            if d["name"] == saved_name:
                return i, saved_name

    try:
        default_idx = sd.default.device[0]
    except Exception:
        default_idx = inputs[0][0]

    clear()
    print(f"\n  {PINK}<(^o^<){RESET}  {BOLD}buddy{RESET}  {DIM}— select microphone{RESET}\n")
    for n, (i, d) in enumerate(inputs, 1):
        marker = f"{GREEN}✦{RESET}" if i == default_idx else f"{DIM}◇{RESET}"
        tag    = f"  {DIM}(system default){RESET}" if i == default_idx else ""
        print(f"  {marker} {BOLD}{n}{RESET}. {d['name']}{tag}")

    print(f"\n  {DIM}number, or Enter for default:{RESET} ", end="", flush=True)
    choice = input().strip()
    if not choice:
        selected = default_idx
    else:
        try:
            selected = inputs[int(choice) - 1][0]
        except (ValueError, IndexError):
            selected = default_idx

    name = sd.query_devices(selected)["name"]
    cfg["device"] = name
    save_config(cfg)
    print(f"\n  {GREEN}✓ using:{RESET} {name}\n")
    time.sleep(0.5)
    return selected, name

# ── stderr suppressor ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def silence_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved   = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)

# ── main ───────────────────────────────────────────────────────────────────────

def run(device_id: int, device_name: str, out_dir: Path, fmt: str,
        hook: Optional[str], model_size: str):

    transcriber = TranscriptionWorker(model_size)
    player      = AudioPlayer()

    # Shared state dict — render thread reads a snapshot under the lock
    lock = threading.Lock()
    st: dict = {
        "mode":          "record",
        "device_name":   device_name,
        "fmt":           fmt,
        "out_dir":       out_dir,
        # record
        "saved":         [],           # [(name, dur)]
        "last_saved":    "",
        "skipped":       False,
        "recording":     False,
        "rec_dur":       0.0,
        "current_rms":   0.0,
        "exit_warning":  False,
        # transcription
        "tx_current":    None,
        "tx_queue_size": 0,
        # browse
        "browse_files":  [],
        "browse_cursor": 0,
        # player
        "player_playing": False,
        "player_current": None,
        # animation
        "tick":          0,
    }

    space_down  = threading.Event()
    space_up    = threading.Event()
    key_q: _queue.Queue = _queue.Queue(maxsize=64)
    should_exit = threading.Event()

    # ── SIGINT: double Ctrl+C ─────────────────────────────────────────────────
    _first_ctrlc: list = [0.0]

    def _sigint(sig, frame):
        now = time.time()
        if transcriber.pending and now - _first_ctrlc[0] > 2.0:
            _first_ctrlc[0] = now
            with lock:
                st["exit_warning"] = True
            return
        should_exit.set()

    signal.signal(signal.SIGINT, _sigint)

    # ── keyboard listener ─────────────────────────────────────────────────────
    def on_press(key):
        mode = st["mode"]  # single read, GIL-safe
        if mode == "record":
            if key == kb.Key.space and not space_down.is_set():
                space_up.clear()
                space_down.set()
            elif getattr(key, "char", None) == "b":
                try:
                    key_q.put_nowait("browse")
                except _queue.Full:
                    pass
        else:
            tag = None
            if key == kb.Key.up:
                tag = "up"
            elif key == kb.Key.down:
                tag = "down"
            elif getattr(key, "char", None) == "p":
                tag = "play"
            elif getattr(key, "char", None) in ("q", "b"):
                tag = "quit_browse"
            if tag:
                try:
                    key_q.put_nowait(tag)
                except _queue.Full:
                    pass

    def on_release(key):
        if key == kb.Key.space:
            space_down.clear()
            space_up.set()

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    with silence_stderr():
        listener.start()
        time.sleep(0.25)

    # ── render thread ─────────────────────────────────────────────────────────
    def _tick_and_sync():
        """Update animated fields in st before render thread copies snapshot."""
        st["tick"] += 1
        st["tx_current"]    = transcriber.current
        st["tx_queue_size"] = transcriber.queue_size
        st["player_playing"] = player.playing
        st["player_current"] = player.current

    def _render():
        while not should_exit.is_set():
            with lock:
                _tick_and_sync()
                snap = dict(st)
                # saved is a list of tuples — copy it
                snap["saved"] = list(st["saved"])
                snap["browse_files"] = list(st["browse_files"])
            if snap["mode"] == "record":
                draw_record(snap)
            else:
                draw_browse(snap)
            time.sleep(0.1)

    threading.Thread(target=_render, daemon=True).start()

    # ── browse file list refresh ──────────────────────────────────────────────
    _last_refresh = [0.0]

    def refresh_browse():
        now = time.time()
        if now - _last_refresh[0] > 2.0:
            files = get_recordings(out_dir)
            with lock:
                st["browse_files"] = files
            _last_refresh[0] = now

    # ── main loop ─────────────────────────────────────────────────────────────
    try:
        while not should_exit.is_set():

            # ── record mode ───────────────────────────────────────────────────
            if st["mode"] == "record":
                # check for key events (browse switch)
                try:
                    ev = key_q.get_nowait()
                    if ev == "browse":
                        player.stop()
                        files = get_recordings(out_dir)
                        with lock:
                            st["browse_files"] = files
                            st["browse_cursor"] = 0
                            st["mode"] = "browse"
                        _last_refresh[0] = time.time()
                    continue
                except _queue.Empty:
                    pass

                if not space_down.wait(timeout=0.05):
                    continue

                # start recording
                chunks: list = []
                start = time.time()
                with lock:
                    st["recording"]    = True
                    st["exit_warning"] = False

                try:
                    with sd.InputStream(device=device_id, samplerate=SAMPLERATE,
                                        channels=CHANNELS, dtype=DTYPE,
                                        blocksize=BLOCKSIZE) as stream:
                        while not space_up.is_set():
                            data, _ = stream.read(BLOCKSIZE)
                            chunks.append(data.copy())
                            with lock:
                                st["current_rms"] = rms(data)
                                st["rec_dur"]     = time.time() - start
                except sd.PortAudioError:
                    with lock:
                        st["recording"] = False
                    space_down.clear()
                    continue

                dur      = time.time() - start
                peak_rms = max((rms(c) for c in chunks), default=0.0)

                with lock:
                    st["recording"] = False

                if dur < MIN_DURATION or peak_rms < SILENCE_THRESHOLD:
                    with lock:
                        st["skipped"] = True
                else:
                    audio    = np.concatenate(chunks, axis=0)
                    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    name     = f"{ts}.{fmt}"
                    out_path = out_dir / name
                    try:
                        write_audio(out_path, audio, fmt)
                        with lock:
                            st["saved"].append((name, dur))
                            st["last_saved"] = f"{name}  ({dur:.1f}s)"
                            st["skipped"]    = False
                        transcriber.enqueue(out_path)
                        run_hook(hook, out_path)
                    except Exception:
                        with lock:
                            st["skipped"] = True

                space_down.clear()

            # ── browse mode ───────────────────────────────────────────────────
            else:
                refresh_browse()

                with lock:
                    cursor = st["browse_cursor"]
                    n_files = len(st["browse_files"])
                if cursor >= n_files:
                    with lock:
                        st["browse_cursor"] = max(0, n_files - 1)

                try:
                    ev = key_q.get(timeout=0.25)
                except _queue.Empty:
                    continue

                if ev == "quit_browse":
                    player.stop()
                    with lock:
                        st["mode"] = "record"

                elif ev == "up":
                    with lock:
                        if st["browse_cursor"] > 0:
                            st["browse_cursor"] -= 1
                    player.stop()

                elif ev == "down":
                    with lock:
                        if st["browse_cursor"] < len(st["browse_files"]) - 1:
                            st["browse_cursor"] += 1
                    player.stop()

                elif ev == "play":
                    with lock:
                        files  = st["browse_files"]
                        cursor = st["browse_cursor"]
                    if files:
                        target = files[cursor]
                        if player.playing and player.current == target:
                            player.stop()
                        else:
                            player.play(target)

    finally:
        should_exit.set()
        listener.stop()
        player.stop()

    # Let render thread finish its last frame before printing
    time.sleep(0.15)
    clear()
    print(f"\n  {DIM}bye.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="buddy",
        description="hold SPACE to record, release to save")
    parser.add_argument("-o", "--output", metavar="DIR", default=".",
                        help="directory for recordings (default: current dir)")
    parser.add_argument("-f", "--format", metavar="FMT", default="wav",
                        choices=FORMATS,
                        help=f"format: {', '.join(FORMATS)}  (default: wav)")
    parser.add_argument("--hook", metavar="CMD",
                        help="run after each save; {file} is replaced with the path")
    parser.add_argument("--model", metavar="SIZE", default="base",
                        help="whisper model size: tiny/base/small/medium  (default: base)")
    parser.add_argument("--pick", action="store_true",
                        help="re-select microphone")
    parser.add_argument("--list", action="store_true",
                        help="list input devices and exit")
    args = parser.parse_args()

    if args.list:
        for i, d in list_inputs():
            print(f"  {i}: {d['name']}")
        return

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    device_id, device_name = pick_device(cfg, force=args.pick)
    run(device_id, device_name, out_dir, args.format, args.hook, args.model)


if __name__ == "__main__":
    main()
