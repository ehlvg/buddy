import argparse
import contextlib
import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

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

# ── colors ─────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
PINK   = "\033[35m"

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

# ── audio ──────────────────────────────────────────────────────────────────────

def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))

def write_audio(path: Path, audio: np.ndarray, fmt: str):
    if fmt == "mp3":
        try:
            import lameenc
        except ImportError:
            raise RuntimeError("mp3 output requires lameenc — pip install lameenc")
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

# ── TUI ────────────────────────────────────────────────────────────────────────

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

def header():
    print(f"\n  {PINK}<(^o^<){RESET}  {BOLD}buddy{RESET}  {DIM}— voice recorder{RESET}\n")

def draw(saved: list, device_name: str, out_dir: Path, fmt: str,
         recording: bool = False, last_saved: str = "",
         rec_dur: float = 0.0, skipped: bool = False, current_rms: float = 0.0):
    clear()
    header()
    print(f"  {DIM}mic:{RESET} {device_name}  "
          f"{DIM}· fmt:{RESET} {fmt}  "
          f"{DIM}· dir:{RESET} {fmt_path(out_dir)}\n")

    if recording:
        print(f"  {RED}● {rec_dur:.1f}s{RESET}  {level_bar(current_rms)}  "
              f"{DIM}release SPACE to save{RESET}\n")
    else:
        print(f"  {DIM}hold SPACE to record  •  Ctrl+C to quit{RESET}\n")

    if not recording:
        if skipped:
            print(f"  {YELLOW}⚠ too quiet — nothing saved{RESET}\n")
        elif last_saved:
            print(f"  {GREEN}✓ saved:{RESET} {last_saved}\n")

    if saved:
        print(f"  {DIM}recent files:{RESET}")
        for fname, dur in saved[-6:]:
            bar_len = min(int(dur * 4), 40)
            print(f"  {DIM}│{RESET} {fname}  {DIM}{'▪' * bar_len} {dur:.1f}s{RESET}")
        print()

# ── device picker ──────────────────────────────────────────────────────────────

def list_inputs() -> list[tuple[int, dict]]:
    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]

def pick_device(cfg: dict, force: bool = False) -> tuple[int, str]:
    inputs = list_inputs()
    if not inputs:
        print(f"  {RED}no input devices found{RESET}")
        sys.exit(1)

    # Use saved device if available and not forced to re-pick
    if not force and (saved_name := cfg.get("device")):
        for i, d in inputs:
            if d["name"] == saved_name:
                return i, saved_name

    try:
        default_idx = sd.default.device[0]
    except Exception:
        default_idx = inputs[0][0]

    clear()
    header()
    print(f"  {DIM}select microphone:{RESET}\n")

    for n, (i, d) in enumerate(inputs, start=1):
        marker      = f"{GREEN}✦{RESET}" if i == default_idx else f"{DIM}◇{RESET}"
        default_tag = f"  {DIM}(system default){RESET}" if i == default_idx else ""
        print(f"  {marker} {BOLD}{n}{RESET}. {d['name']}{default_tag}")

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

# ── main loop ──────────────────────────────────────────────────────────────────

def run(device_id: int, device_name: str, out_dir: Path, fmt: str):
    saved      = []
    last_saved = ""
    skipped    = False

    space_down = threading.Event()
    space_up   = threading.Event()

    def on_press(key):
        if key == kb.Key.space and not space_down.is_set():
            space_up.clear()
            space_down.set()

    def on_release(key):
        if key == kb.Key.space:
            space_down.clear()
            space_up.set()

    draw(saved, device_name, out_dir, fmt)

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    with silence_stderr():
        listener.start()
        time.sleep(0.25)

    try:
        while True:
            if not space_down.wait(timeout=0.05):
                continue

            chunks      = []
            start       = time.time()
            last_draw   = 0.0
            current_rms = 0.0

            try:
                with sd.InputStream(device=device_id, samplerate=SAMPLERATE,
                                    channels=CHANNELS, dtype=DTYPE,
                                    blocksize=BLOCKSIZE) as stream:
                    while not space_up.is_set():
                        data, _ = stream.read(BLOCKSIZE)
                        chunks.append(data.copy())
                        current_rms = rms(data)
                        now = time.time()
                        if now - last_draw >= 0.1:
                            draw(saved, device_name, out_dir, fmt,
                                 recording=True, rec_dur=now - start,
                                 current_rms=current_rms)
                            last_draw = now
            except sd.PortAudioError as e:
                print(f"\n  {RED}microphone error: {e}{RESET}\n", file=sys.stderr)
                space_down.clear()
                continue

            dur      = time.time() - start
            peak_rms = max((rms(c) for c in chunks), default=0.0)

            if dur < MIN_DURATION or peak_rms < SILENCE_THRESHOLD:
                skipped = True
            else:
                audio    = np.concatenate(chunks, axis=0)
                ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                name     = f"{ts}.{fmt}"
                out_path = out_dir / name
                try:
                    write_audio(out_path, audio, fmt)
                    saved.append((name, dur))
                    last_saved = f"{name}  ({dur:.1f}s)"
                    skipped    = False
                except Exception as e:
                    print(f"\n  {RED}save error: {e}{RESET}\n", file=sys.stderr)
                    skipped = True

            draw(saved, device_name, out_dir, fmt,
                 last_saved=last_saved, skipped=skipped)
            space_down.clear()

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()

    print(f"\n  {DIM}bye.{RESET}\n")

# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="buddy",
                                     description="hold SPACE to record, release to save")
    parser.add_argument("-o", "--output", metavar="DIR", default=".",
                        help="directory to save recordings (default: current dir)")
    parser.add_argument("-f", "--format", metavar="FMT", default="wav",
                        choices=FORMATS,
                        help=f"output format: {', '.join(FORMATS)}  (default: wav)")
    parser.add_argument("--pick", action="store_true",
                        help="force microphone selection prompt")
    parser.add_argument("--list", action="store_true",
                        help="list available input devices and exit")
    args = parser.parse_args()

    if args.list:
        for i, d in list_inputs():
            print(f"  {i}: {d['name']}")
        return

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    device_id, device_name = pick_device(cfg, force=args.pick)
    run(device_id, device_name, out_dir, args.format)


if __name__ == "__main__":
    main()
