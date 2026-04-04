import os
import sys
import time
import uuid
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard as kb

# ── audio ──────────────────────────────────────────────────────────────────────

SAMPLERATE = 44100
CHANNELS   = 1
DTYPE      = "int16"
BLOCKSIZE  = 1024

# ── TUI helpers ────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"


def clear():
    os.system("cls" if sys.platform == "win32" else "clear")


def draw(saved: list, recording: bool = False, last_saved: str = "", rec_dur: float = 0.0):
    clear()
    print(f"\n  {BOLD}buddy{RESET}  {DIM}— voice recorder{RESET}\n")

    if recording:
        print(f"  {RED}● recording...  {rec_dur:.1f}s{RESET}  {DIM}release SPACE to save{RESET}\n")
    else:
        print(f"  {DIM}hold SPACE to record  •  Ctrl+C to quit{RESET}\n")

    if last_saved and not recording:
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
    saved = []
    last_saved = ""

    space_down = threading.Event()
    space_up   = threading.Event()
    quit_event = threading.Event()

    def on_press(key):
        if key == kb.Key.space:
            # Ignore OS key-repeat: only act on the first press
            if not space_down.is_set():
                space_up.clear()
                space_down.set()

    def on_release(key):
        if key == kb.Key.space:
            space_down.clear()
            space_up.set()

    draw(saved)

    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            while not quit_event.is_set():
                # Wait for spacebar to be pressed
                if not space_down.wait(timeout=0.05):
                    continue

                # ── begin recording ────────────────────────────────────────
                space_up.clear()
                chunks = []
                stream = sd.InputStream(
                    samplerate=SAMPLERATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    blocksize=BLOCKSIZE,
                )
                stream.start()
                start = time.time()
                last_draw = 0.0

                while not space_up.is_set():
                    data, _ = stream.read(BLOCKSIZE)
                    chunks.append(data.copy())
                    now = time.time()
                    if now - last_draw >= 0.1:          # redraw at ~10 fps
                        draw(saved, recording=True, rec_dur=now - start)
                        last_draw = now

                stream.stop()
                stream.close()
                # ── end recording ──────────────────────────────────────────

                dur   = time.time() - start
                audio = np.concatenate(chunks, axis=0)

                name = uuid.uuid4().hex[:8] + ".wav"
                sf.write(name, audio, SAMPLERATE)

                saved.append((name, dur))
                last_saved = f"{name}  ({dur:.1f}s)"

                draw(saved, recording=False, last_saved=last_saved)

                space_down.clear()

        except KeyboardInterrupt:
            pass

    print(f"\n  {DIM}bye.{RESET}\n")


def main():
    run()


if __name__ == "__main__":
    main()
