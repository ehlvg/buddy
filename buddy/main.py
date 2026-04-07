import sys
import time
import secrets
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard as kb

# ── audio ──────────────────────────────────────────────────────────────────────

SAMPLERATE        = 44100
CHANNELS          = 1
DTYPE             = "int16"
BLOCKSIZE         = 1024
MIN_DURATION      = 0.3     # seconds — discard accidental taps
SILENCE_THRESHOLD = 300     # peak RMS for int16 (max 32767); below = silence

# ── TUI helpers ────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
PINK   = "\033[35m"
CYAN   = "\033[36m"

METER_WIDTH = 20


def clear():
    print("\033[2J\033[H", end="", flush=True)


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))


def level_bar(value: float, max_val: float = 3000.0) -> str:
    filled = int(min(value / max_val, 1.0) * METER_WIDTH)
    empty  = METER_WIDTH - filled
    color  = RED if filled > METER_WIDTH * 0.75 else GREEN
    return f"{color}{'█' * filled}{DIM}{'░' * empty}{RESET}"


def header():
    print(f"\n  {PINK}<(^o^<){RESET}  {BOLD}buddy{RESET}  {DIM}— voice recorder{RESET}\n")


# ── device picker ──────────────────────────────────────────────────────────────

def pick_device() -> int:
    inputs = [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]

    if not inputs:
        print(f"  {RED}no input devices found{RESET}")
        sys.exit(1)

    try:
        default_idx = sd.default.device[0]
    except Exception:
        default_idx = inputs[0][0]

    # ── draw picker ────────────────────────────────────────────────────────────
    clear()
    header()
    print(f"  {DIM}select microphone:{RESET}\n")

    for n, (i, d) in enumerate(inputs, start=1):
        marker = f"{GREEN}✦{RESET}" if i == default_idx else f"{DIM}◇{RESET}"
        default_tag = f"  {DIM}(system default){RESET}" if i == default_idx else ""
        print(f"  {marker} {BOLD}{n}{RESET}. {d['name']}{default_tag}")

    print(f"\n  {DIM}enter number, or press Enter for default:{RESET} ", end="", flush=True)

    choice = input().strip()

    if not choice:
        selected = default_idx
    else:
        try:
            n = int(choice) - 1
            selected = inputs[n][0]
        except (ValueError, IndexError):
            selected = default_idx

    name = sd.query_devices(selected)["name"]
    print(f"\n  {GREEN}✓ using:{RESET} {name}\n")
    time.sleep(0.6)
    return selected


# ── main draw ──────────────────────────────────────────────────────────────────

def draw(saved: list, device_name: str, recording: bool = False,
         last_saved: str = "", rec_dur: float = 0.0,
         skipped: bool = False, current_rms: float = 0.0):
    clear()
    header()

    print(f"  {DIM}mic:{RESET} {device_name}\n")

    if recording:
        bar = level_bar(current_rms)
        print(f"  {RED}● {rec_dur:.1f}s{RESET}  {bar}  {DIM}release SPACE to save{RESET}\n")
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
            bar = "▪" * bar_len
            print(f"  {DIM}│{RESET} {fname}  {DIM}{bar} {dur:.1f}s{RESET}")
        print()


# ── main loop ──────────────────────────────────────────────────────────────────

def run():
    device_id   = pick_device()
    device_name = sd.query_devices(device_id)["name"]

    saved      = []
    last_saved = ""
    skipped    = False

    space_down = threading.Event()
    space_up   = threading.Event()

    def on_press(key):
        if key == kb.Key.space:
            if not space_down.is_set():
                space_up.clear()   # cleared atomically with space_down.set — fixes race
                space_down.set()

    def on_release(key):
        if key == kb.Key.space:
            space_down.clear()
            space_up.set()

    draw(saved, device_name)

    with kb.Listener(on_press=on_press, on_release=on_release):
        try:
            while True:
                if not space_down.wait(timeout=0.05):
                    continue

                # ── begin recording ────────────────────────────────────────
                chunks      = []
                start       = time.time()
                last_draw   = 0.0
                current_rms = 0.0

                try:
                    with sd.InputStream(
                        device=device_id,
                        samplerate=SAMPLERATE,
                        channels=CHANNELS,
                        dtype=DTYPE,
                        blocksize=BLOCKSIZE,
                    ) as stream:
                        while not space_up.is_set():
                            data, _ = stream.read(BLOCKSIZE)
                            chunks.append(data.copy())
                            current_rms = rms(data)
                            now = time.time()
                            if now - last_draw >= 0.1:
                                draw(saved, device_name, recording=True,
                                     rec_dur=now - start, current_rms=current_rms)
                                last_draw = now
                except sd.PortAudioError as e:
                    print(f"\n  {RED}microphone error: {e}{RESET}\n", file=sys.stderr)
                    space_down.clear()
                    continue
                # ── end recording ──────────────────────────────────────────

                dur      = time.time() - start
                peak_rms = max((rms(c) for c in chunks), default=0.0)

                if dur < MIN_DURATION or peak_rms < SILENCE_THRESHOLD:
                    skipped = True
                else:
                    audio = np.concatenate(chunks, axis=0)
                    name  = secrets.token_hex(4) + ".wav"
                    sf.write(name, audio, SAMPLERATE)
                    saved.append((name, dur))
                    last_saved = f"{name}  ({dur:.1f}s)"
                    skipped    = False

                draw(saved, device_name, recording=False,
                     last_saved=last_saved, skipped=skipped)
                space_down.clear()

        except KeyboardInterrupt:
            pass

    print(f"\n  {DIM}bye.{RESET}\n")


def main():
    run()


if __name__ == "__main__":
    main()
