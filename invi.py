"""
InvisINK — Air-Drawing Math Solver  (InvisINK Edition)
=======================================================
Draw mathematical expressions in the air using hand gestures,
then solve them with a thumb-up. Uses MediaPipe for hand tracking
and InvisINK Vision AI for math recognition & solving
(no TensorFlow / CNN required).

Gestures
--------
- INDEX finger only   → Draw
- INDEX + MIDDLE      → Erase
- THUMB UP            → Solve  (sends canvas to InvisINK)
- OPEN HAND           → Clear canvas
- PINKY only          → Cycle history
- Q key               → Quit
- E key               → Export snapshot
- M key               → Toggle voice mute
- D key               → Delete history

"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import warnings
import yaml

# ═══════════════════════════════════════════════════════════════════
#  CONFIG LOADING
# ═══════════════════════════════════════════════════════════════════

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    """Load configuration from YAML file, with fallback defaults."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning("config.yaml not found, using defaults.")
        return {}


CFG = load_config()

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS (loaded from config.yaml)
# ═══════════════════════════════════════════════════════════════════

_draw_cfg = CFG.get("drawing", {})
BRUSH_THICKNESS: int  = _draw_cfg.get("brush_thickness", 14)
ERASER_RADIUS: int    = _draw_cfg.get("eraser_radius", 28)
GLOW_BLUR_KERNEL: tuple[int, int] = tuple(_draw_cfg.get("glow_blur_kernel", [19, 19]))
GLOW_BLEND_ALPHA: float = _draw_cfg.get("glow_blend_alpha", 0.55)

_flash = CFG.get("solve_flash", {})
SOLVE_FLASH_FRAMES: int    = _flash.get("frames", 14)
SOLVE_FLASH_MAX_ALPHA: float = _flash.get("max_alpha", 0.28)
SOLVE_FLASH_BRIGHTNESS: int  = _flash.get("brightness", 240)

_ui = CFG.get("ui", {})
TOP_PANEL_HEIGHT: int   = _ui.get("top_panel_height", 110)
BOTTOM_BAR_HEIGHT: int  = _ui.get("bottom_bar_height", 54)
BADGE_RECT: tuple[int, int, int, int] = tuple(_ui.get("badge_rect", [16, 12, 176, 52]))
MAX_EXPR_CHARS: int     = _ui.get("max_expr_chars", 72)
MAX_SOL_CHARS: int      = _ui.get("max_sol_chars", 68)
CAMERA_BLEND_ALPHA: float = _ui.get("camera_blend_alpha", 0.35)

_mp_cfg = CFG.get("mediapipe", {})
MAX_HANDS: int             = _mp_cfg.get("max_hands", 2)
DETECTION_CONFIDENCE: float = _mp_cfg.get("detection_confidence", 0.80)
TRACKING_CONFIDENCE: float  = _mp_cfg.get("tracking_confidence", 0.70)

_fps_cfg = CFG.get("fps", {})
FPS_SMOOTHING: float = _fps_cfg.get("smoothing", 0.9)
FPS_INITIAL: float   = _fps_cfg.get("initial", 30.0)

_hist_cfg = CFG.get("history", {})
HISTORY_MAX_DISPLAY: int = _hist_cfg.get("max_display", 5)
HISTORY_FILE: str        = _hist_cfg.get("file", "history.json")

_export_cfg = CFG.get("export", {})
EXPORT_DIR: str = _export_cfg.get("directory", "exports")

_voice_cfg = CFG.get("voice", {})
VOICE_ENABLED: bool = _voice_cfg.get("enabled", True)
VOICE_RATE: int     = _voice_cfg.get("rate", 160)

# -- InvisINK API config ---------------------------------------------
_gemini_cfg = CFG.get("gemini", {})

# ── All API keys (auto-rotated when quota is hit) ──────────────────
from dotenv import load_dotenv
load_dotenv()

_API_KEYS: list[str] = [
    k.strip() for k in
    os.environ.get("GEMINI_API_KEYS", "").split(",")
    if k.strip()
]
_API_KEYS = list(dict.fromkeys(_API_KEYS))  # preserve order, remove duplicates

GEMINI_MODEL: str            = _gemini_cfg.get("model", "gemini-3-flash-preview")
GEMINI_MIN_CONTOUR_AREA: int = _gemini_cfg.get("min_contour_area", 50)

# -- Colors (from config or defaults) ---
_colors_cfg = CFG.get("colors", {})
COLOR = {
    "neon_draw":    tuple(_colors_cfg.get("neon_draw",    [0, 240, 160])),
    "neon_glow":    tuple(_colors_cfg.get("neon_glow",    [0, 200, 120])),
    "accent_cyan":  tuple(_colors_cfg.get("accent_cyan",  [255, 220, 0])),
    "accent_gold":  tuple(_colors_cfg.get("accent_gold",  [0, 200, 255])),
    "success":      tuple(_colors_cfg.get("success",      [80, 220, 100])),
    "text_dim":     tuple(_colors_cfg.get("text_dim",     [130, 130, 160])),
    "danger":       tuple(_colors_cfg.get("danger",       [80, 80, 240])),
    "erase":        tuple(_colors_cfg.get("erase",        [180, 100, 255])),
    "overlay_dark": tuple(_colors_cfg.get("overlay_dark", [8, 8, 14])),
    "api_badge":    (0, 180, 255),   # extra: teal badge for "API" mode indicator
}

_mode_colors_cfg = CFG.get("mode_colors", {})
MODE_COLORS = {
    "DRAW":    tuple(_mode_colors_cfg.get("DRAW",    [0, 240, 160])),
    "ERASE":   tuple(_mode_colors_cfg.get("ERASE",   [180, 100, 255])),
    "SOLVE":   tuple(_mode_colors_cfg.get("SOLVE",   [0, 200, 255])),
    "CLEAR":   tuple(_mode_colors_cfg.get("CLEAR",   [80, 80, 240])),
    "IDLE":    tuple(_mode_colors_cfg.get("IDLE",    [130, 130, 160])),
    "WAITING": (0, 180, 255),   # InvisINK API call in progress
}

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX

# -- Logging ----------------------------------------------------------
logger = logging.getLogger("InvisINK-API")

# ═══════════════════════════════════════════════════════════════════
#  InvisINK AI CLIENT (lazy import — graceful fallback if not installed)
# ═══════════════════════════════════════════════════════════════════


class ApiKeyManager:
    """Rotate through a pool of API keys when quota is exhausted."""

    # Errors that indicate a quota / rate-limit hit
    _QUOTA_SIGNALS = ("429", "quota", "exhausted", "resource_exhausted",
                      "rateLimitExceeded", "too many requests")

    def __init__(self, keys: list[str]) -> None:
        self._keys   = keys
        self._index  = 0
        self._lock   = threading.Lock()

    # ── public api ──────────────────────────────────────────────────

    @property
    def current_key(self) -> str:
        return self._keys[self._index] if self._keys else ""

    @property
    def total(self) -> int:
        return len(self._keys)

    @property
    def index(self) -> int:
        return self._index

    def is_quota_error(self, exc: Exception) -> bool:
        """Return True if *exc* looks like a rate-limit / quota error."""
        msg = str(exc).lower()
        return any(sig in msg for sig in self._QUOTA_SIGNALS)

    def rotate(self) -> bool:
        """Advance to the next key. Returns True if a new key is available."""
        with self._lock:
            if self._index + 1 < len(self._keys):
                self._index += 1
                logger.warning(
                    "Quota hit — switched to API key %d / %d.",
                    self._index + 1, len(self._keys),
                )
                return True
            logger.error(
                "All %d API keys exhausted. Try again later.",
                len(self._keys),
            )
            return False

    def configure(self) -> bool:
        """Configure google.generativeai with the current key. Returns success."""
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.current_key)
            return True
        except Exception as exc:
            logger.warning("Could not configure genai: %s", exc)
            return False

    def reset(self) -> None:
        """Reset to the first key (call at app start or after a long idle)."""
        with self._lock:
            self._index = 0


# ── Instantiate & initial configure ────────────────────────────────

_gemini_available = False
_key_manager      = ApiKeyManager(_API_KEYS)

try:
    import google.generativeai as genai  # type: ignore
    if _key_manager.current_key:
        _key_manager.configure()
        _gemini_available = True
        logger.info(
            "InvisINK API ready. Key 1/%d. Model: %s",
            _key_manager.total, GEMINI_MODEL,
        )
    else:
        logger.warning(
            "InvisINK API key not set. "
            "Set env var GEMINI_API_KEY or add gemini.api_key in config.yaml."
        )
except ImportError:
    logger.warning(
        "google-generativeai not installed. "
        "Run: pip install google-generativeai"
    )


# ═══════════════════════════════════════════════════════════════════
#  VOICE ENGINE (background thread)
# ═══════════════════════════════════════════════════════════════════

class VoiceFeedback:
    """Speak answers aloud using pyttsx3 in a background thread."""

    def __init__(self, enabled: bool = VOICE_ENABLED, rate: int = VOICE_RATE) -> None:
        self.enabled = enabled
        self.muted   = False
        self._rate   = rate
        self._lock   = threading.Lock()

    def speak(self, text: str) -> None:
        """Speak *text* in a daemon thread (non-blocking)."""
        if not self.enabled or self.muted:
            return
        t = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
        t.start()

    def _speak_sync(self, text: str) -> None:
        with self._lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                logger.warning("Voice error: %s", e)

    def toggle_mute(self) -> bool:
        """Toggle mute state. Returns new muted status."""
        self.muted = not self.muted
        return self.muted


# ═══════════════════════════════════════════════════════════════════
#  HISTORY MANAGER
# ═══════════════════════════════════════════════════════════════════

class HistoryManager:
    """Track, persist, and display solved expressions."""

    def __init__(
        self,
        filepath: str = HISTORY_FILE,
        max_display: int = HISTORY_MAX_DISPLAY,
    ) -> None:
        self.filepath    = Path(filepath)
        self.max_display = max_display
        self.entries: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        """Load history from JSON file if it exists."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
                logger.info("Loaded %d history entries.", len(self.entries))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load history: %s", e)
                self.entries = []

    def save(self) -> None:
        """Persist history to JSON file."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d history entries.", len(self.entries))
        except OSError as e:
            logger.warning("Could not save history: %s", e)

    def add(self, expression: str, solution: str) -> None:
        """Add a new solved expression to history."""
        if not expression or not solution:
            return
        self.entries.append({
            "expression": expression,
            "solution":   solution,
            "time":       datetime.now().strftime("%H:%M:%S"),
        })

    def get_recent(self) -> list[dict[str, str]]:
        """Return the most recent entries for display."""
        return self.entries[-self.max_display:]

    def clear_history(self) -> None:
        """Delete all history entries and overwrite the JSON file."""
        self.entries = []
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump([], f)
            logger.info("History cleared.")
        except OSError as e:
            logger.warning("Could not clear history file: %s", e)


# ═══════════════════════════════════════════════════════════════════
#  EXPORT MANAGER
# ═══════════════════════════════════════════════════════════════════

class ExportManager:
    """Save canvas snapshots with result overlay to disk."""

    def __init__(self, export_dir: str = EXPORT_DIR) -> None:
        self.export_path = Path(export_dir)
        self.export_path.mkdir(exist_ok=True)

    def export_snapshot(
        self,
        display: np.ndarray,
        expression: str,
        solution: str,
    ) -> str:
        """Save a snapshot and return the filepath."""
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"invisink_api_{stamp}.png"
        filepath = self.export_path / filename

        img    = display.copy()
        h, w   = img.shape[:2]
        cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"Expr: {expression}", (20, h - 50),
                    FONT_MONO, 0.6, COLOR["accent_cyan"], 1, cv2.LINE_AA)
        cv2.putText(img, f"Ans:  {solution}", (20, h - 20),
                    FONT_MONO, 0.7, COLOR["success"], 1, cv2.LINE_AA)

        cv2.imwrite(str(filepath), img)
        logger.info("Exported snapshot: %s", filepath)
        return str(filepath)


# ═══════════════════════════════════════════════════════════════════
#  InvisINK MATH SOLVER
# ═══════════════════════════════════════════════════════════════════

_GEMINI_PROMPT = (
    "You are a math OCR + solver assistant. The image shows a handwritten "
    "mathematical expression drawn on a dark background in bright neon strokes. "
    "1. Identify the mathematical expression exactly as written. "
    "2. Solve it (arithmetic, algebra, calculus — whatever is shown). "
    "3. Reply with ONLY two lines:\n"
    "   EXPR: <the expression you recognised>\n"
    "   ANS: <the answer>\n"
    "Do NOT add explanations, markdown, or any other text."
)


def canvas_to_png_bytes(canvas: np.ndarray) -> bytes:
    """Convert a grayscale drawing canvas to PNG bytes for the API."""
    # Build a white-on-black BGR image for clarity
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    # Tint the strokes in neon green so they look like the HUD
    tinted = np.zeros_like(bgr)
    tinted[canvas > 0] = [0, 240, 160]
    _, buf = cv2.imencode(".png", tinted)
    return buf.tobytes()


def solve_with_gemini(canvas: np.ndarray) -> tuple[str, str]:
    """Send the drawing canvas to InvisINK Vision AI and return (expression, answer).

    Automatically rotates through all API keys on quota / rate-limit errors.

    Returns
    -------
    tuple[str, str]
        ``(recognised_expression, solution_string)``
    """
    # Quick check: is there anything drawn?
    if cv2.countNonZero(canvas) < GEMINI_MIN_CONTOUR_AREA:
        return "", "Draw something first"

    if not _gemini_available:
        return "", "InvisINK not available — check the connectivity"

    import google.generativeai as genai  # type: ignore

    png_bytes  = canvas_to_png_bytes(canvas)
    image_part = {
        "inline_data": {
            "mime_type": "image/png",
            "data": base64.b64encode(png_bytes).decode("utf-8"),
        }
    }

    # ── Retry loop — one attempt per available API key ───────────────
    attempts = _key_manager.total
    for attempt in range(attempts):
        _key_manager.configure()          # apply current key
        model = genai.GenerativeModel(GEMINI_MODEL)
        try:
            response = model.generate_content([_GEMINI_PROMPT, image_part])
            raw      = response.text.strip()
            logger.debug("InvisINK raw response (key %d):\n%s",
                         _key_manager.index + 1, raw)

            # Parse the two-line response
            expr = ""
            ans  = ""
            for line in raw.splitlines():
                line = line.strip()
                if line.upper().startswith("EXPR:"):
                    expr = line.split(":", 1)[1].strip()
                elif line.upper().startswith("ANS:"):
                    ans = line.split(":", 1)[1].strip()

            if not expr and not ans:
                return "?", raw[:MAX_SOL_CHARS]

            return expr or "?", ans or "No answer"

        except Exception as exc:
            if _key_manager.is_quota_error(exc):
                logger.warning(
                    "Key %d/%d quota hit (%s) — trying next key…",
                    _key_manager.index + 1, _key_manager.total, type(exc).__name__,
                )
                if not _key_manager.rotate():
                    return "", "All API keys exhausted — try again later"
                # loop continues with next key
            else:
                logger.exception("InvisINK solve error (key %d)",
                                 _key_manager.index + 1)
                return "", f"InvisINK Error: {exc}"

    return "", "All API keys exhausted — try again later"


# ═══════════════════════════════════════════════════════════════════
#  UI DRAWING HELPERS  (identical to new.py)
# ═══════════════════════════════════════════════════════════════════

def draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int  = 12,
    thickness: int = -1,
    alpha: float = 0.6,
) -> None:
    """Draw a rounded rectangle with alpha blending onto *img* in-place."""
    x1, y1 = pt1
    x2, y2 = pt2
    ov = img.copy()
    cv2.rectangle(ov, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(ov, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [
        (x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius), (x2 - radius, y2 - radius),
    ]:
        cv2.circle(ov, (cx, cy), radius, color, thickness)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_circle_glow(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled circle with a soft glow halo onto *img* in-place."""
    for i in range(3, 0, -1):
        ov = img.copy()
        cv2.circle(ov, center, radius + i * 6, color, 2)
        cv2.addWeighted(ov, 0.12 * i / 3, img, 1 - 0.12 * i / 3, 0, img)
    cv2.circle(img, center, radius, color, -1)


def put_text_shadow(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Put text with a dark drop-shadow for readability on any background."""
    cv2.putText(img, text, (pos[0] + 2, pos[1] + 2), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def truncate_text(text: str, max_chars: int = MAX_EXPR_CHARS) -> str:
    """Return *text* truncated with an ellipsis if it exceeds *max_chars*."""
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"


# ═══════════════════════════════════════════════════════════════════
#  FINGER STATE DETECTOR  (identical to new.py)
# ═══════════════════════════════════════════════════════════════════

def fingers_up(lm: Any, hand_label: str) -> list[bool]:
    """Return a 5-element list indicating which fingers are raised.

    Index mapping: ``[thumb, index, middle, ring, pinky]``.
    Thumb detection is mirrored based on *hand_label*.
    """
    tips = [4, 8, 12, 16, 20]
    f: list[bool] = []

    # Thumb — compare tip x vs. IP joint x (mirrored for left hand)
    t_tip = lm.landmark[tips[0]].x
    t_ip  = lm.landmark[tips[0] - 1].x
    f.append(t_tip < t_ip if hand_label == "Right" else t_tip > t_ip)

    # Other fingers — tip y above PIP joint y
    for i in range(1, 5):
        f.append(lm.landmark[tips[i]].y < lm.landmark[tips[i] - 2].y)

    return f


# ═══════════════════════════════════════════════════════════════════
#  HUD RENDERER  (identical look to new.py — one extra "API" badge)
# ═══════════════════════════════════════════════════════════════════

def render_hud(
    display: np.ndarray,
    mode: str,
    last_expression: str,
    last_solution: str,
    handedness_str: str,
    history: list[dict[str, str]] | None = None,
    voice_muted: bool = False,
    api_waiting: bool = False,
) -> None:
    """Draw the full heads-up display: top panel, badges, history, and bottom bar."""
    h, w      = display.shape[:2]
    mode_col  = MODE_COLORS.get(mode, COLOR["text_dim"])

    # ── Top panel background ──────────────────────────
    ov = display.copy()
    cv2.rectangle(ov, (0, 0), (w, TOP_PANEL_HEIGHT), COLOR["overlay_dark"], -1)
    cv2.addWeighted(ov, 0.82, display, 0.18, 0, display)

    # Accent separator line + glow
    cv2.line(display, (0, TOP_PANEL_HEIGHT), (w, TOP_PANEL_HEIGHT), mode_col, 2)
    gv = display.copy()
    cv2.line(gv, (0, TOP_PANEL_HEIGHT), (w, TOP_PANEL_HEIGHT), mode_col, 8)
    cv2.addWeighted(gv, 0.20, display, 0.80, 0, display)

    # ── Mode badge (top-left) ─────────────────────────
    bx1, by1, bx2, by2 = BADGE_RECT
    draw_rounded_rect(display, (bx1, by1), (bx2, by2), mode_col, radius=8, alpha=0.18)
    cv2.rectangle(display, (bx1, by1), (bx2, by2), mode_col, 1)
    badge_text = "WAITING…" if api_waiting else mode
    put_text_shadow(display, badge_text, (30, 41), FONT, 0.66, mode_col)

    # ── "InvisINK" engine badge (top-left, second row) ─
    engine_col = COLOR["api_badge"]
    draw_rounded_rect(display, (bx1, by2 + 4), (bx2, by2 + 26),
                      engine_col, radius=6, alpha=0.15)
    cv2.rectangle(display, (bx1, by2 + 4), (bx2, by2 + 26), engine_col, 1)
    put_text_shadow(display, "InvisINK", (22, by2 + 20), FONT_MONO, 0.38, engine_col)

    # ── Handedness (top-right) ────────────────────────
    hand_text = f"{handedness_str} Hand"
    (htw, _), _ = cv2.getTextSize(hand_text, FONT_MONO, 0.6, 1)
    put_text_shadow(display, hand_text, (w - htw - 20, 38), FONT_MONO, 0.6,
                    COLOR["accent_gold"])

    # ── Voice mute indicator ──────────────────────────
    if voice_muted:
        put_text_shadow(display, "MUTED", (w - htw - 20, 58),
                        FONT_MONO, 0.4, COLOR["danger"])

    # ── ROW 2 : Expression ───────────────────────────
    expr_str = truncate_text(last_expression) if last_expression else "—"
    put_text_shadow(display, "expr:", (20, 76), FONT_MONO, 0.48, COLOR["text_dim"])
    put_text_shadow(display, expr_str, (76, 76), FONT_MONO, 0.60, COLOR["accent_cyan"])

    # ── ROW 3 : Solution ─────────────────────────────
    if api_waiting:
        sol_str = "Asking InvisINK…"
        sol_col = COLOR["api_badge"]
    elif last_solution:
        sol_str = truncate_text(last_solution, MAX_SOL_CHARS)
        sol_col = COLOR["success"]
    else:
        sol_str = "draw expression and show THUMB UP to solve"
        sol_col = COLOR["text_dim"]

    put_text_shadow(display, "ans:", (20, 100), FONT_MONO, 0.48, COLOR["text_dim"])
    put_text_shadow(display, sol_str, (76, 100), FONT_MONO, 0.60, sol_col)

    # ── History panel (right side) ────────────────────
    if history:
        hist_x       = w - 320
        hist_y_start = TOP_PANEL_HEIGHT + 20

        ov2 = display.copy()
        cv2.rectangle(
            ov2,
            (hist_x - 10, hist_y_start - 10),
            (w - 10, hist_y_start + len(history) * 40 + 10),
            COLOR["overlay_dark"], -1,
        )
        cv2.addWeighted(ov2, 0.75, display, 0.25, 0, display)

        put_text_shadow(display, "HISTORY", (hist_x, hist_y_start + 5),
                        FONT_MONO, 0.45, COLOR["accent_gold"])

        for i, entry in enumerate(reversed(history)):
            y_pos  = hist_y_start + 25 + i * 40
            expr_t = truncate_text(entry.get("expression", ""), 28)
            sol_t  = truncate_text(entry.get("solution",   ""), 28)
            time_t = entry.get("time", "")

            put_text_shadow(display, f"{time_t} {expr_t}", (hist_x, y_pos),
                            FONT_MONO, 0.38, COLOR["text_dim"])
            put_text_shadow(display, f"  → {sol_t}", (hist_x, y_pos + 16),
                            FONT_MONO, 0.38, COLOR["success"])

    # ── Bottom hint bar ───────────────────────────────
    bov = display.copy()
    cv2.rectangle(bov, (0, h - BOTTOM_BAR_HEIGHT), (w, h),
                  COLOR["overlay_dark"], -1)
    cv2.addWeighted(bov, 0.85, display, 0.15, 0, display)
    cv2.line(display, (0, h - BOTTOM_BAR_HEIGHT), (w, h - BOTTOM_BAR_HEIGHT),
             (40, 40, 56), 1)

    hints = [
        ("INDEX",       "Draw",          COLOR["neon_draw"]),
        ("INDEX+MID",   "Erase",         COLOR["erase"]),
        ("THUMB UP",    "Solve (InvisINK)", COLOR["accent_gold"]),
        ("OPEN HAND",   "Clear",         COLOR["danger"]),
        ("D",           "Del History",   COLOR["danger"]),
        ("E / M / Q",   "Exp/Mute/Quit", COLOR["text_dim"]),
    ]
    slot_w = w // len(hints)
    for i, (key, label, col) in enumerate(hints):
        cx = i * slot_w + slot_w // 2
        (kw, _), _ = cv2.getTextSize(key,   FONT_MONO, 0.48, 1)
        (lw, _), _ = cv2.getTextSize(label, FONT_MONO, 0.40, 1)
        put_text_shadow(display, key,   (cx - kw // 2, h - 33), FONT_MONO, 0.48, col)
        put_text_shadow(display, label, (cx - lw // 2, h - 14), FONT_MONO, 0.40,
                        COLOR["text_dim"])
    for i in range(1, len(hints)):
        cv2.line(display, (i * slot_w, h - BOTTOM_BAR_HEIGHT + 6),
                 (i * slot_w, h - 6), (40, 40, 56), 1)


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════════

class InvisINKApiApp:
    """InvisINK Edition.

    Identical gesture loop and HUD to InvisINKApp in new.py,
    but replaces the TensorFlow CNN solver with an InvisINK Vision AI call.
    """

    def __init__(self) -> None:
        warnings.filterwarnings("ignore")

        # -- Subsystems ----------------------------------------
        self.history  = HistoryManager()
        self.exporter = ExportManager()
        self.voice    = VoiceFeedback()

        # -- Drawing state -------------------------------------
        self.px:   int  = 0
        self.py:   int  = 0
        self.submit_lock:          bool = False
        self.clear_lock:           bool = False
        self.last_expression:      str  = ""
        self.last_solution:        str  = ""
        self.mode:                 str  = "IDLE"
        self.solve_flash:          int  = 0
        self.delete_history_flash: int  = 0
        self.fps:   float = FPS_INITIAL
        self.fps_t: float = time.time()

        # -- Async API call state ------------------------------
        self._api_lock:                threading.Lock = threading.Lock()
        self._api_waiting:             bool           = False
        self._pending_expression:      str            = ""
        self._pending_solution:        str            = ""
        self._api_result_ready:        bool           = False

        logger.info("InvisINK Edition — System Online ✦")
        if not _gemini_available:
            logger.warning(
                "No InvisINK key found. "
                "Set InvisINK env variable or add to config.yaml."
            )

    # ── gesture handlers ───────────────────────────────────────

    def _handle_draw(self, ix: int, iy: int, canvas: np.ndarray) -> None:
        """Handle the DRAW gesture (index finger only)."""
        self.mode = "DRAW"
        if self.px:
            cv2.line(canvas, (self.px, self.py), (ix, iy), 255, BRUSH_THICKNESS)
        self.px, self.py = ix, iy
        self.submit_lock = False
        self.clear_lock  = False

    def _handle_erase(
        self, ix: int, iy: int,
        canvas: np.ndarray, display: np.ndarray,
    ) -> None:
        """Handle the ERASE gesture (index + middle fingers)."""
        self.mode = "ERASE"
        cv2.circle(canvas,  (ix, iy), ERASER_RADIUS, 0,              -1)
        cv2.circle(display, (ix, iy), ERASER_RADIUS, COLOR["erase"],  2)
        self.px = self.py = 0

    def _handle_solve(self, canvas: np.ndarray, display: np.ndarray) -> None:
        """Handle the SOLVE gesture (thumb up only) — async InvisINK call."""
        if self._api_waiting:
            return  # already in progress

        self.mode        = "WAITING"
        self.submit_lock = True
        self.solve_flash = SOLVE_FLASH_FRAMES

        # Snapshot the canvas NOW (before next frame erases it)
        canvas_copy = canvas.copy()
        display_copy = display.copy()

        def _call():
            expr, ans = solve_with_gemini(canvas_copy)
            with self._api_lock:
                self._pending_expression = expr
                self._pending_solution   = ans
                self._api_result_ready   = True

        with self._api_lock:
            self._api_waiting      = True
            self._api_result_ready = False

        t = threading.Thread(target=_call, daemon=True)
        t.start()

        # Store display copy for export once result arrives
        self._display_for_export = display_copy

    def _handle_clear(self, canvas: np.ndarray) -> None:
        """Handle the CLEAR gesture (all fingers open)."""
        self.mode = "CLEAR"
        canvas[:] = 0
        self.last_expression = ""
        self.last_solution   = ""
        self.clear_lock  = True
        self.submit_lock = False
        self.px = self.py = 0

    def _poll_api_result(self) -> None:
        """Check if the async InvisINK call has returned and update state."""
        with self._api_lock:
            if self._api_waiting and self._api_result_ready:
                expr = self._pending_expression
                ans  = self._pending_solution
                self._api_waiting      = False
                self._api_result_ready = False
            else:
                return

        self.last_expression = expr
        self.last_solution   = ans
        self.mode            = "SOLVE"
        self.solve_flash     = SOLVE_FLASH_FRAMES

        # Persist & voice
        self.history.add(expr, ans)
        self.voice.speak(f"Answer is: {ans}")

        # Export snapshot
        if hasattr(self, "_display_for_export"):
            self.exporter.export_snapshot(
                self._display_for_export, expr, ans,
            )

        logger.info("InvisINK result → expr: %s  ans: %s", expr, ans)

    # ── main loop ─────────────────────────────────────────────

    def run(self) -> None:
        """Open the camera and enter the main gesture-processing loop."""
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils

        hands = mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

        skel_lm   = mp_draw.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=4)
        skel_conn = mp_draw.DrawingSpec(color=(0, 180, 255), thickness=2)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.critical("Cannot open camera (index 0). Check your webcam connection.")
            sys.exit(1)

        sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        drawing_canvas = np.zeros((sh, sw), dtype=np.uint8)

        logger.info("Press Q inside the window to quit.\n")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    logger.warning("Failed to read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands.process(rgb)

                # ── Camera background ────────────────────────
                display = cv2.addWeighted(
                    frame, CAMERA_BLEND_ALPHA,
                    np.zeros_like(frame), 0, 0,
                )

                detected_handedness = "Right"

                if res.multi_hand_landmarks:
                    lm        = res.multi_hand_landmarks[0]
                    hand_info = res.multi_handedness[0]
                    detected_handedness = hand_info.classification[0].label

                    mp_draw.draw_landmarks(
                        display, lm, mp_hands.HAND_CONNECTIONS,
                        skel_lm, skel_conn,
                    )

                    f  = fingers_up(lm, detected_handedness)
                    ix = int(lm.landmark[8].x * sw)
                    iy = int(lm.landmark[8].y * sh)

                    draw_circle_glow(display, (ix, iy), 10, COLOR["neon_draw"])

                    # ── Gesture dispatch ──────────────
                    if f == [False, True, False, False, False]:
                        self._handle_draw(ix, iy, drawing_canvas)

                    elif f == [False, True, True, False, False]:
                        self._handle_erase(ix, iy, drawing_canvas, display)

                    elif f[0] and not any(f[1:]) and not self.submit_lock:
                        self._handle_solve(drawing_canvas, display)

                    elif all(f) and not self.clear_lock:
                        self._handle_clear(drawing_canvas)

                    else:
                        self.px = self.py = 0
                        if not self.submit_lock and not self._api_waiting:
                            self.mode = "IDLE"
                else:
                    self.px = self.py = 0
                    if not self.submit_lock and not self._api_waiting:
                        self.mode = "IDLE"

                # ── Poll async InvisINK result ────────
                self._poll_api_result()

                # ── Neon drawing composite ────────────
                glow  = np.zeros_like(display)
                glow[drawing_canvas > 0] = COLOR["neon_glow"]
                blurred = cv2.GaussianBlur(glow, GLOW_BLUR_KERNEL, 0)
                display = cv2.addWeighted(display, 1.0, blurred, GLOW_BLEND_ALPHA, 0)
                display[drawing_canvas > 0] = COLOR["neon_draw"]

                # ── Solve flash ───────────────────────
                if self.solve_flash > 0:
                    a     = self.solve_flash / SOLVE_FLASH_FRAMES * SOLVE_FLASH_MAX_ALPHA
                    flash = np.full_like(display, SOLVE_FLASH_BRIGHTNESS)
                    display = cv2.addWeighted(display, 1 - a, flash, a, 0)
                    self.solve_flash -= 1

                # ── Waiting spinner overlay ───────────
                if self._api_waiting:
                    spin_char = "◐◓◑◒"[int(time.time() * 4) % 4]
                    put_text_shadow(
                        display,
                        f"{spin_char} Asking InvisINK…",
                        (sw // 2 - 120, sh // 2),
                        FONT, 1.0, COLOR["api_badge"], 2,
                    )

                # ── HUD ───────────────────────────────
                render_hud(
                    display, self.mode,
                    self.last_expression, self.last_solution,
                    detected_handedness,
                    history=self.history.get_recent(),
                    voice_muted=self.voice.muted,
                    api_waiting=self._api_waiting,
                )

                # ── FPS counter ───────────────────────
                now = time.time()
                self.fps = (
                    FPS_SMOOTHING * self.fps
                    + (1 - FPS_SMOOTHING) / max(now - self.fps_t, 1e-5)
                )
                self.fps_t = now
                put_text_shadow(
                    display,
                    f"{self.fps:.0f} fps",
                    (sw - 80, TOP_PANEL_HEIGHT - 14),
                    FONT_MONO, 0.45, COLOR["text_dim"],
                )

                cv2.imshow("InvisINK", display)

                # ── "History Deleted" flash notice ────
                if self.delete_history_flash > 0:
                    notice = "History Deleted!"
                    (nw, nh), _ = cv2.getTextSize(notice, FONT, 0.9, 2)
                    nx = (sw - nw) // 2
                    ny = sh // 2
                    draw_rounded_rect(
                        display,
                        (nx - 20, ny - nh - 14), (nx + nw + 20, ny + 10),
                        COLOR["danger"], radius=10, alpha=0.72,
                    )
                    put_text_shadow(display, notice, (nx, ny), FONT, 0.9,
                                    (255, 255, 255), 2)
                    self.delete_history_flash -= 1

                # ── Keyboard handling ─────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("e"):
                    path = self.exporter.export_snapshot(
                        display, self.last_expression, self.last_solution,
                    )
                    logger.info("Manual export: %s", path)
                elif key == ord("m"):
                    muted = self.voice.toggle_mute()
                    logger.info("Voice %s", "muted" if muted else "unmuted")
                elif key == ord("d"):
                    self.history.clear_history()
                    self.delete_history_flash = 60
                    logger.info("History deleted by user.")

        finally:
            self.history.save()
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Session ended.")


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    """Configure logging and launch the InvisINK API application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = InvisINKApiApp()
    app.run()


if __name__ == "__main__":
    main()
