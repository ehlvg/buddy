<img src="logo.svg" width="110" alt="buddy">

# buddy

hold **space** to record · release to save · ctrl+c to quit

saves as `<hex>.wav` in your current directory.

---

## install

```bash
pipx install .
```

## usage

```bash
buddy
```

---

## requirements

- python 3.9+
- a microphone
- `libportaudio` — `brew install portaudio` on mac, `apt install libportaudio2` on linux, nothing on windows

> **macos:** grant accessibility permission on first run — system settings → privacy → accessibility

## uninstall

```bash
pipx uninstall buddy-recorder
```
