# buddy

Hold **SPACE** to record. Release to save. Repeat. Ctrl+C to quit.

Files are saved as `<8-char-hex>.wav` in your **current directory**.

---

## Install

```bash
pip install .
```

## Usage

```bash
buddy
```

---

## Requirements

- Python 3.9+
- A working microphone
- `libportaudio` (for `sounddevice`)

### macOS
```bash
brew install portaudio
```

> **Accessibility permission required.** On first run macOS will prompt you to grant Terminal (or your app) access under **System Settings → Privacy & Security → Accessibility**.

### Linux (Debian/Ubuntu)
```bash
sudo apt install libportaudio2
```

### Windows
No extra steps — portaudio ships with the `sounddevice` wheel.

---

## Uninstall

```bash
pip uninstall buddy-recorder
```
