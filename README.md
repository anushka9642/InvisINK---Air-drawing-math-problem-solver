# InvisINK---Air-drawing-math-problem-solver
InvisINK is a gesture-controlled, air-drawing math solver. Use your index finger as a pen in front of your webcam, draw any mathematical expression mid-air, and flash a thumbs-up to have Google Gemini recognize and solve it instantly  complete with voice readout, solve history, and snapshot exports

# ✍️ InvisINK

### Air-Drawing Math Solver

**Draw math expressions in the air with your finger. Solve them with a thumbs-up.**

---

## 🌟 What is InvisINK?

InvisINK is a real-time, gesture-controlled math solver that uses your **webcam and bare hand** as the input device. Point your index finger at the camera and write math in the air — InvisINK tracks your fingertip using MediaPipe, renders your strokes on a neon-glow canvas overlay, and sends the drawing to **Google Gemini Vision AI** to recognize and solve the expression. The answer is displayed on a HUD and read aloud via text-to-speech.

No stylus. No touchscreen. No machine learning training required.

---

## 🎥 Demo

> Draw `2x² + 3x - 5 = 0` in the air → 👍 → *"x = 1 or x = −2.5"*

---

## ✨ Features

- **✋ Hand Gesture Control** — Five distinct gestures mapped to draw, erase, solve, clear, and browse history
- **🤖 Gemini Vision AI** — Sends your canvas to Google Gemini for OCR + math solving; no custom CNN needed
- **🌈 Neon Glow Canvas** — Real-time neon-stroke rendering blended over the live webcam feed
- **🔊 Voice Readout** — Text-to-speech announces the solved answer (toggleable)
- **📜 Session History** — Last 5 solved expressions shown on the HUD; persisted to `history.json`
- **📸 Snapshot Export** — Export the current canvas + solution as a timestamped PNG
- **⚙️ YAML Configuration** — Every parameter (colors, thresholds, brush size, voice rate) in one `config.yaml`
- **🔄 API Key Rotation** — Automatically rotates Gemini API keys on quota errors
- **📊 Live FPS Counter** — Smoothed real-time frame rate display

---

## ✋ Gestures

| Gesture | Action |
|---|---|
| ☝️ **Index finger only** | Draw — trace your math expression |
| ✌️ **Index + Middle fingers** | Erase — rub out strokes |
| 👍 **Thumb up** | Solve — sends canvas to Gemini AI |
| 🖐️ **Open hand (all fingers)** | Clear — wipe the entire canvas |
| 🤙 **Pinky only** | Cycle — browse solve history |
| `Q` key | Quit the application |
| `E` key | Export snapshot to `exports/` |
| `M` key | Toggle voice mute on/off |
| `D` key | Delete all history |

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10** or higher
- A working **webcam**
- A free **Google Gemini API key** — [get one here](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/InvisINK.git
cd InvisINK
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key (safe for GitHub)

Copy `.env.example` to `.env`:

```bash
# macOS / Linux
cp .env.example .env

# Windows (PowerShell)
copy .env.example .env
```

Then set your key in `.env`:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

Or add multiple keys for auto-rotation:

```env
GEMINI_API_KEYS=key1,key2,key3
```

> 🔒 **Never commit your real API key.** `.env` is already ignored by `.gitignore`.

### 4. Run

```bash
python invi.py
```

Your webcam window will open. Hold your hand in frame and start drawing!

---

## ⚙️ Configuration

All settings live in `config.yaml`. No code changes needed for common tweaks.

| Section | Key Options |
|---|---|
| `drawing` | `brush_thickness`, `eraser_radius`, `glow_blend_alpha` |
| `mediapipe` | `max_hands`, `detection_confidence`, `tracking_confidence` |
| `gemini` | `model`, `min_contour_area` |
| `voice` | `enabled` (true/false), `rate` (words per minute) |
| `colors` | HUD colors in BGR format (`neon_draw`, `accent_cyan`, `success`, etc.) |
| `history` | `max_display` (entries shown on HUD), `file` (JSON path) |
| `export` | `directory` (output folder for snapshots) |
| `fps` | `smoothing` (EMA weight), `initial` (starting FPS estimate) |

### Example tweak — thicker brush, slower voice:

```yaml
drawing:
  brush_thickness: 20

voice:
  rate: 130
```

---

## 📁 Project Structure

```
InvisINK/
├── invi.py            # Main application — gesture engine, HUD, API calls
├── config.yaml        # All configurable parameters
├── requirements.txt   # Python dependencies
├── history.json       # Auto-generated session history
├── exports/           # Auto-generated snapshot PNGs
└── README.md
```

---

## 🧠 How It Works

```
Webcam Frame
     │
     ▼
MediaPipe Hand Tracking
     │  landmark positions (21 keypoints)
     ▼
Gesture Classifier  (fingers_up logic)
     │  DRAW / ERASE / SOLVE / CLEAR / HISTORY
     ▼
Drawing Canvas  (grayscale overlay)
     │  neon glow via Gaussian blur + alpha blend
     ▼
  [on THUMB UP]
     │
     ▼
canvas_to_png_bytes()  →  Gemini Vision AI
                               │  OCR + solve
                               ▼
                     expression + solution
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               HUD Display          Voice Readout
          (OpenCV overlay)           (pyttsx3)
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python-headless` | Webcam capture, canvas rendering, image processing |
| `mediapipe` | Real-time hand landmark detection |
| `google-generativeai` | Gemini Vision AI — math OCR and solving |
| `numpy` | Array operations on the drawing canvas |
| `pyttsx3` | Offline text-to-speech voice readout |
| `pyyaml` | Config file loading |
| `sympy` | (Optional) local math fallback |

---

## 🔐 API Key Safety

- Store Gemini keys in `.env` only — **do not push secrets to GitHub**
- The `.gitignore` excludes `.env` files by default
- Use `.env.example` as your shareable template (safe to commit)
- InvisINK supports **multiple API keys** with automatic rotation using `GEMINI_API_KEYS=key1,key2,key3`

---

## 🗺️ Roadmap

- [ ] Streamlit web interface (browser-based, no local install)
- [ ] Multi-line expression support
- [ ] Step-by-step solution breakdown
- [ ] Custom gesture remapping via config
- [ ] LaTeX output rendering

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a pull request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

Built with ☕ + 🤚 + AI — *because calculators are too mainstream*

