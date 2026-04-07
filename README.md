<img alt="image" src="shot.png" />

<img src="logo.svg" width="110" alt="buddy">

# buddy

hold **space** to record · release to save · **b** to browse recordings · ctrl+c to quit

saves timestamped files (`2026-04-07_14-30-22.wav`) to your current directory, or wherever you point it.
transcribes each recording automatically in the background (if `faster-whisper` is installed).

---

## install

```bash
pipx install .
```

with transcription support:

```bash
pipx install '.[transcribe]'
```

## usage

```bash
buddy                        # wav to current dir, picks mic on first run
buddy -o ~/recordings        # save to a specific folder
buddy -f mp3                 # save as mp3 (also: flac, ogg)
buddy -f ogg -o ~/voice      # combine flags
buddy --pick                 # re-select microphone
buddy --list                 # list available input devices
buddy --model small          # use a larger whisper model (tiny/base/small/medium)
buddy --hook "cp {file} ~/dropbox/"   # run a command after each save
buddy --hook "whisper {file}"         # transcribe with an external tool instead
```

### browse mode

press **b** to open the recording browser:

- **UP / DOWN** — navigate the list (stops playback)
- **p** — play or stop the selected file
- **q** — back to recording

transcripts (if available) are shown below each filename in gray.

### transcription

after each recording, buddy queues it for local transcription via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
a spinner shows progress; `.txt` files are saved next to the audio.

if a transcription is in progress when you press Ctrl+C, buddy warns you.
press Ctrl+C **again within 2 seconds** to force-quit.

### post-save hooks

`--hook "cmd {file}"` — buddy runs this after each successful save.
`{file}` is replaced with the full path to the new recording. runs in the background.

```bash
buddy --hook "say 'saved'"
buddy --hook "rsync {file} user@host:~/voice/"
```

mic choice is remembered between runs (`~/.config/buddy/config.json`).

---

## requirements

- python 3.9+
- a microphone
- `libportaudio` — `brew install portaudio` on mac, `apt install libportaudio2` on linux, nothing on windows
- `faster-whisper` (optional) — for local transcription: `pipx inject buddy-recorder faster-whisper`

> **macos:** grant accessibility permission on first run — system settings → privacy → accessibility

## uninstall

```bash
pipx uninstall buddy-recorder
```
