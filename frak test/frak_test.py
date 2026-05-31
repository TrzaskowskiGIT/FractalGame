""" GPU-accelerated fractal exploration application.

The application renders Mandelbrot, Julia, and Burning Ship fractals using CUDA compute kernels executed through CuPy.

Primary systems:

GPU fractal rendering
Double-double precision viewport math
Interactive camera navigation
Runtime palette switching
Export rendering pipeline
Menu-driven UI system
Julia fractal parameter editing

The architecture separates rendering, input handling, menu rendering, camera control, and application state management into isolated systems. """

# ============================================================================
# IMPORTS
# ============================================================================

import json
import os
import sys
import time

from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

try:
    import cupy as cp
except Exception:
    print("ERROR: CUDA/CuPy not installed")
    sys.exit(1)

try:
    import pygame as pg
except Exception:
    print("ERROR: PyGame not installed")
    sys.exit(1)

try:
    import numpy as np
except Exception:
    print("ERROR: NumPy not installed")
    sys.exit(1)

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class AppConfig:

    """ Stores global application configuration.
    The configuration object centralizes:
    - window sizing
    - rendering resolution
    - export resolution
    - frame limiting
    - UI font configuration
    - camera movement scaling
    
    Runtime systems reference this shared configuration
    object instead of duplicating constants.
    """

    upscale: int = 1

    base_width: int = 1920
    base_height: int = 1080

    png_export_width: int = 3840
    png_export_height: int = 2160

    gif_export_width: int = 1280
    gif_export_height: int = 720

    target_fps: int = 60

    base_move_speed: float = 0.05

    music_volume: float = 0.15
    ui_volume: float = 0.5
    show_performance_info: bool = False

    font_name: str = "consolas"
    font_size: int = 24

    window_title: str = "GPU Fractal Explorer"

    @property
    def width(self):
        return int(self.base_width * self.upscale)

    @property
    def height(self):
        return int(self.base_height * self.upscale)

    @property
    def export_width(self):
        return self.png_export_width

    @export_width.setter
    def export_width(self, value):
        self.png_export_width = int(value)

    @property
    def export_height(self):
        return self.png_export_height

    @export_height.setter
    def export_height(self, value):
        self.png_export_height = int(value)


CONFIG = AppConfig()

PARAMETER_EDIT_LOCK_WIDTH = 1e-6

BENCHMARK_WIDTH = 640
BENCHMARK_HEIGHT = 360
BENCHMARK_RUNS = 3

GIF_FRAME_COUNT = 90
GIF_DURATION_MS = 40
GIF_ZOOM_FACTOR = 0.92
GIF_MAX_WIDTH = 1920
GIF_MAX_HEIGHT = 1080

GAME_RESOLUTION_PRESETS = [
    (1280, 720),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
]

PNG_RESOLUTION_PRESETS = [
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
    (7680, 4320),
]

GIF_RESOLUTION_PRESETS = [
    (640, 360),
    (854, 480),
    (1280, 720),
    (1920, 1080),
]

USER_SETTINGS_DIR = "settings"
USER_SETTINGS_FILE = os.path.join(
    USER_SETTINGS_DIR,
    "user_settings.json"
)

USER_PROGRESS_FILE = os.path.join(
    USER_SETTINGS_DIR,
    "user_progress.json"
)

EXPORT_DIR = "exports"

PNG_EXPORT_DIR = os.path.join(
    EXPORT_DIR,
    "pictures"
)

GIF_EXPORT_DIR = os.path.join(
    EXPORT_DIR,
    "gifs"
)

ACHIEVEMENTS = {
    "tutorial_complete": (
        "First Steps",
        "Complete the tutorial."
    ),
    "first_zoom": (
        "Into the Fractal",
        "Zoom in or out for the first time."
    ),
    "palette_swapper": (
        "Color Explorer",
        "Change the color palette."
    ),
    "fractal_switcher": (
        "Shape Shifter",
        "Switch to a different fractal type."
    ),
    "first_export": (
        "Fractal Photographer",
        "Export your first PNG or GIF."
    ),
    "deep_zoom": (
        "Deep Diver",
        "Zoom deeply enough for parameter editing to become locked."
    ),
    "benchmark_runner": (
        "Speed Tester",
        "Run the GPU vs CPU benchmark."
    ),
}

TUTORIAL_STEPS = [
    {
        "title": "Welcome",
        "text": "This is a GPU fractal explorer rendered through CUDA/CuPy.",
        "action": "begin",
        "required": "Press Enter to begin.",
    },
    {
        "title": "Movement",
        "text": "Move the camera around the fractal.",
        "action": "movement",
        "required": "Press W, A, S, or D.",
    },
    {
        "title": "Zooming",
        "text": "Zoom in or out and watch the viewport update.",
        "action": "zoom",
        "required": "Use the mouse wheel.",
    },
    {
        "title": "Centering",
        "text": "Re-center the camera on a point in the fractal view.",
        "action": "center",
        "required": "Left-click the fractal view.",
    },
    {
        "title": "Iterations",
        "text": "Change the iteration count used by the renderer.",
        "action": "iterations",
        "required": "Press Q or E.",
    },
    {
        "title": "Palette",
        "text": "Switch to another color palette.",
        "action": "palette",
        "required": "Press R or F.",
    },
    {
        "title": "Fractal Parameters",
        "text": "Edit Mandelbrot or Julia parameters when available.",
        "action": "parameters",
        "required": "Press I, J, K, or L.",
    },
    {
        "title": "Menu",
        "text": "Don't forget to explore other fractals and settings under the ECS menu.",
        "action": "menu",
        "required": "Press ESC to complete the tutorial.",
    },
]


def default_user_settings():

    return {
        "game_width": 1920,
        "game_height": 1080,
        "png_export_width": 3840,
        "png_export_height": 2160,
        "gif_export_width": 1280,
        "gif_export_height": 720,
        "music_volume": 0.15,
        "ui_volume": 0.5,
        "show_performance_info": False,
    }


def clamp_volume(value):

    try:
        value = float(value)
    except Exception:
        value = 0.0

    return max(0.0, min(1.0, value))


def normalize_resolution(width, height, presets, default):

    try:
        width = int(width)
        height = int(height)
    except Exception:
        return default

    if (width, height) in presets:
        return width, height

    return default


def normalize_user_settings(data):

    defaults = default_user_settings()

    if not isinstance(data, dict):
        data = {}

    game_width, game_height = normalize_resolution(
        data.get("game_width"),
        data.get("game_height"),
        GAME_RESOLUTION_PRESETS,
        (defaults["game_width"], defaults["game_height"])
    )

    png_width, png_height = normalize_resolution(
        data.get("png_export_width"),
        data.get("png_export_height"),
        PNG_RESOLUTION_PRESETS,
        (defaults["png_export_width"], defaults["png_export_height"])
    )

    gif_width, gif_height = normalize_resolution(
        data.get("gif_export_width"),
        data.get("gif_export_height"),
        GIF_RESOLUTION_PRESETS,
        (defaults["gif_export_width"], defaults["gif_export_height"])
    )

    return {
        "game_width": game_width,
        "game_height": game_height,
        "png_export_width": png_width,
        "png_export_height": png_height,
        "gif_export_width": gif_width,
        "gif_export_height": gif_height,
        "music_volume": clamp_volume(
            data.get("music_volume", defaults["music_volume"])
        ),
        "ui_volume": clamp_volume(
            data.get("ui_volume", defaults["ui_volume"])
        ),
        "show_performance_info": (
            data.get(
                "show_performance_info",
                defaults["show_performance_info"]
            )
            if isinstance(
                data.get(
                    "show_performance_info",
                    defaults["show_performance_info"]
                ),
                bool
            )
            else defaults["show_performance_info"]
        ),
    }


def save_user_settings(settings):

    os.makedirs(
        USER_SETTINGS_DIR,
        exist_ok=True
    )

    with open(USER_SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            settings,
            file,
            indent=4
        )


def load_user_settings():

    os.makedirs(
        USER_SETTINGS_DIR,
        exist_ok=True
    )

    if not os.path.exists(USER_SETTINGS_FILE):
        settings = default_user_settings()
        save_user_settings(settings)
        return settings

    try:
        with open(USER_SETTINGS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

    except Exception as e:
        print(f"WARNING: Failed to load user settings: {e}")
        data = {}

    settings = normalize_user_settings(data)
    save_user_settings(settings)
    return settings


def apply_user_settings_to_config(settings):

    CONFIG.upscale = 1
    CONFIG.base_width = settings["game_width"]
    CONFIG.base_height = settings["game_height"]
    CONFIG.png_export_width = settings["png_export_width"]
    CONFIG.png_export_height = settings["png_export_height"]
    CONFIG.gif_export_width = settings["gif_export_width"]
    CONFIG.gif_export_height = settings["gif_export_height"]
    CONFIG.music_volume = settings["music_volume"]
    CONFIG.ui_volume = settings["ui_volume"]
    CONFIG.show_performance_info = settings["show_performance_info"]


def current_user_settings():

    return {
        "game_width": CONFIG.base_width,
        "game_height": CONFIG.base_height,
        "png_export_width": CONFIG.png_export_width,
        "png_export_height": CONFIG.png_export_height,
        "gif_export_width": CONFIG.gif_export_width,
        "gif_export_height": CONFIG.gif_export_height,
        "music_volume": CONFIG.music_volume,
        "ui_volume": CONFIG.ui_volume,
        "show_performance_info": CONFIG.show_performance_info,
    }


def update_and_save_user_setting(key, value):

    settings = current_user_settings()
    settings[key] = value
    settings = normalize_user_settings(settings)
    apply_user_settings_to_config(settings)
    save_user_settings(settings)
    return settings


def save_current_user_settings():

    save_user_settings(
        normalize_user_settings(
            current_user_settings()
        )
    )


def default_user_progress():

    return {
        "tutorial_completed": False,
        "achievements": {
            achievement_id: False
            for achievement_id in ACHIEVEMENTS
        }
    }


def normalize_user_progress(data):

    defaults = default_user_progress()

    if not isinstance(data, dict):
        data = {}

    achievements = data.get("achievements", {})

    if not isinstance(achievements, dict):
        achievements = {}

    return {
        "tutorial_completed": (
            data.get(
                "tutorial_completed",
                defaults["tutorial_completed"]
            )
            if isinstance(data.get("tutorial_completed", False), bool)
            else defaults["tutorial_completed"]
        ),
        "achievements": {
            achievement_id: bool(
                achievements.get(achievement_id, False)
            )
            for achievement_id in ACHIEVEMENTS
        }
    }


def save_user_progress(progress):

    os.makedirs(
        USER_SETTINGS_DIR,
        exist_ok=True
    )

    with open(USER_PROGRESS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            normalize_user_progress(progress),
            file,
            indent=4
        )


def load_user_progress():

    os.makedirs(
        USER_SETTINGS_DIR,
        exist_ok=True
    )

    if not os.path.exists(USER_PROGRESS_FILE):
        progress = default_user_progress()
        save_user_progress(progress)
        return progress

    try:
        with open(USER_PROGRESS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

    except Exception as e:
        print(f"WARNING: Failed to load user progress: {e}")
        data = {}

    progress = normalize_user_progress(data)
    save_user_progress(progress)
    return progress


def unlock_achievement(state, achievement_id):

    if achievement_id not in ACHIEVEMENTS:
        return

    achievements = state.user_progress.setdefault(
        "achievements",
        {}
    )

    if achievements.get(achievement_id, False):
        return

    achievements[achievement_id] = True
    save_user_progress(state.user_progress)

    name = ACHIEVEMENTS[achievement_id][0]

    state.export_message = (
        f"Achievement unlocked: {name}"
    )

    state.export_message_timer = time.time()


def complete_tutorial(state):

    state.tutorial_active = False
    state.tutorial_step_index = 0

    if not state.user_progress.get("tutorial_completed", False):
        state.user_progress["tutorial_completed"] = True
        save_user_progress(state.user_progress)

    unlock_achievement(state, "tutorial_complete")


def skip_tutorial(state):

    state.tutorial_active = False
    state.tutorial_step_index = 0
    state.user_progress["tutorial_completed"] = True
    save_user_progress(state.user_progress)


def advance_tutorial(state):

    if state.tutorial_step_index >= len(TUTORIAL_STEPS) - 1:
        complete_tutorial(state)
        return

    state.tutorial_step_index += 1


def get_tutorial_step(state):

    step_index = max(
        0,
        min(state.tutorial_step_index, len(TUTORIAL_STEPS) - 1)
    )

    return TUTORIAL_STEPS[step_index]


def get_tutorial_action(state):

    return get_tutorial_step(state).get(
        "action",
        ""
    )


def is_tutorial_parameter_step_available(state):

    return (
        state.current_fractal in (
            FractalType.MANDELBROT,
            FractalType.JULIA
        )
        and not is_parameter_edit_locked(state)
    )


def get_tutorial_step_text(state):

    step = get_tutorial_step(state)
    text = step.get("text", "")
    required = step.get("required", "")

    if step.get("action") == "parameters":

        if state.current_fractal == FractalType.BURNING_SHIP:
            text = (
                "Burning Ship parameters are not editable in this explorer."
            )
            required = "Press Enter to continue."

        elif is_parameter_edit_locked(state):
            text = (
                "Parameter editing is locked at this deep zoom level."
            )
            required = "Press Enter to continue."

    return text, required


def handle_tutorial_action(state, action_id, audio_manager=None):

    expected_action = get_tutorial_action(state)

    if expected_action == "parameters":

        if not is_tutorial_parameter_step_available(state):
            expected_action = "parameter_continue"

    if action_id != expected_action:
        return False

    if expected_action == "finish":
        state.menu_open = False
        state.submenu_open = None

    advance_tutorial(state)

    if audio_manager:
        audio_manager.play_click()

    return True


def check_deep_zoom_achievement(state):

    if is_parameter_edit_locked(state):
        unlock_achievement(state, "deep_zoom")

# ============================================================================
# AUDIO
# ============================================================================


class AudioManager:

    """ Centralizes application audio playback.
    The audio manager owns mixer initialization, asset loading,
    background music playback, and short UI sound effects.
    Missing files or unavailable audio devices are reported as
    warnings so the renderer and UI can continue running.
    """

    MUSIC_PATH = os.path.join(
        "assets",
        "audio",
        "music",
        "background_music.ogg"
    )

    HOVER_PATH = os.path.join(
        "assets",
        "audio",
        "ui",
        "menu_hover.wav"
    )

    CLICK_PATH = os.path.join(
        "assets",
        "audio",
        "ui",
        "menu_click.wav"
    )

    def __init__(self):

        self.enabled = False
        self.hover_sound = None
        self.click_sound = None

        try:
            if not pg.mixer.get_init():
                pg.mixer.init()

            self.enabled = True

        except Exception as e:
            print(f"WARNING: Audio disabled: {e}")
            return

        self.hover_sound = self.load_sound(
            self.HOVER_PATH
        )

        self.click_sound = self.load_sound(
            self.CLICK_PATH
        )

        if self.hover_sound:
            self.hover_sound.set_volume(
                CONFIG.ui_volume
            )

        if self.click_sound:
            self.click_sound.set_volume(
                CONFIG.ui_volume
            )

    def load_sound(self, path):

        if not self.enabled:
            return None

        if not os.path.exists(path):
            print(f"WARNING: Missing audio file: {path}")
            return None

        try:
            return pg.mixer.Sound(path)

        except Exception as e:
            print(f"WARNING: Failed to load audio file {path}: {e}")
            return None

    def start_music(self):

        if not self.enabled:
            return

        if not os.path.exists(self.MUSIC_PATH):
            print(f"WARNING: Missing audio file: {self.MUSIC_PATH}")
            return

        try:
            pg.mixer.music.load(
                self.MUSIC_PATH
            )

            pg.mixer.music.set_volume(
                CONFIG.music_volume
            )

            pg.mixer.music.play(-1)

        except Exception as e:
            print(f"WARNING: Failed to start background music: {e}")

    def set_music_volume(self, volume):

        CONFIG.music_volume = clamp_volume(volume)

        if self.enabled:
            pg.mixer.music.set_volume(
                CONFIG.music_volume
            )

    def set_ui_volume(self, volume):

        CONFIG.ui_volume = clamp_volume(volume)

        if self.hover_sound:
            self.hover_sound.set_volume(
                CONFIG.ui_volume
            )

        if self.click_sound:
            self.click_sound.set_volume(
                CONFIG.ui_volume
            )

    def play_hover(self):

        if self.hover_sound:
            self.hover_sound.play()

    def play_click(self):

        if self.click_sound:
            self.click_sound.play()


# ============================================================================
# ENUMS
# ============================================================================


class FractalType(str, Enum):

    """ Enumerates all supported fractal rendering modes.
    The integer ordering used by the CUDA kernel is mapped
    from these enum values inside the rendering pipeline.
    """

    MANDELBROT = "mandelbrot"
    JULIA = "julia"
    BURNING_SHIP = "burning_ship"


# ============================================================================
# JULIA PRESETS
# ============================================================================

JULIA_PRESETS = [
    ("Classic", -0.8, 0.156),
    ("Spiral", -0.4, 0.6),
    ("Dendrite", 0.285, 0.01),
    ("Lightning", -0.70176, -0.3842),
    ("Cloud", -0.835, -0.2321),
]

# ============================================================================
# MANDELBROT PRESETS
# ============================================================================

MANDELBROT_PRESETS = [
    ("Classic", 0.0, 0.0),
    ("Real Offset", -0.25, 0.0),
    ("Imaginary Offset", 0.0, 0.25),
    ("Diagonal Offset", -0.15, 0.15),
    ("Experimental", 0.3, -0.2),
]

POWER_VALUES = [
    2,
    3,
    4,
    5,
    6,
]

# ============================================================================
# USER PRESETS
# ============================================================================

USER_PRESET_DIR = "presets"
USER_PRESET_FILE = os.path.join(
    USER_PRESET_DIR,
    "user_presets.json"
)

def empty_user_presets():

    return {
        "julia": [],
        "mandelbrot": [],
    }


def load_user_presets():

    os.makedirs(
        USER_PRESET_DIR,
        exist_ok=True
    )

    if not os.path.exists(USER_PRESET_FILE):
        presets = empty_user_presets()
        save_user_presets(presets)
        return presets

    try:
        with open(USER_PRESET_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

    except Exception as e:
        print(f"WARNING: Failed to load user presets: {e}")
        return empty_user_presets()

    presets = empty_user_presets()

    if not isinstance(data, dict):
        return presets

    for preset in data.get("julia", []):

        if not isinstance(preset, dict):
            continue

        try:
            name = str(preset["name"]).strip()
            cx = float(preset["julia_cx"])
            cy = float(preset["julia_cy"])
        except Exception:
            continue

        if not name:
            continue

        normalized = {
            "name": name,
            "julia_cx": cx,
            "julia_cy": cy,
        }

        if preset.get("fractal_power") in POWER_VALUES:
            normalized["fractal_power"] = int(preset["fractal_power"])

        presets["julia"].append(normalized)

    for preset in data.get("mandelbrot", []):

        if not isinstance(preset, dict):
            continue

        try:
            name = str(preset["name"]).strip()
            zx = float(preset["mandelbrot_zx"])
            zy = float(preset["mandelbrot_zy"])
        except Exception:
            continue

        if not name:
            continue

        normalized = {
            "name": name,
            "mandelbrot_zx": zx,
            "mandelbrot_zy": zy,
        }

        if preset.get("fractal_power") in POWER_VALUES:
            normalized["fractal_power"] = int(preset["fractal_power"])

        presets["mandelbrot"].append(normalized)

    return presets


def save_user_presets(custom_presets):

    os.makedirs(
        USER_PRESET_DIR,
        exist_ok=True
    )

    with open(USER_PRESET_FILE, "w", encoding="utf-8") as file:
        json.dump(
            custom_presets,
            file,
            indent=4
        )


def get_combined_julia_presets(state):

    presets = []

    for name, cx, cy in JULIA_PRESETS:
        presets.append({
            "name": name,
            "julia_cx": cx,
            "julia_cy": cy,
            "custom": False,
        })

    for preset in state.custom_presets.get("julia", []):
        combined = preset.copy()
        combined["custom"] = True
        presets.append(combined)

    return presets


def get_combined_mandelbrot_presets(state):

    presets = []

    for name, zx, zy in MANDELBROT_PRESETS:
        presets.append({
            "name": name,
            "mandelbrot_zx": zx,
            "mandelbrot_zy": zy,
            "custom": False,
        })

    for preset in state.custom_presets.get("mandelbrot", []):
        combined = preset.copy()
        combined["custom"] = True
        presets.append(combined)

    return presets


def apply_julia_preset(state, preset, index):

    state.julia_preset_index = index
    state.julia_preset_name = preset["name"]

    state.julia_cx = preset["julia_cx"]
    state.julia_cy = preset["julia_cy"]

    if preset.get("fractal_power") in POWER_VALUES:
        state.fractal_power = preset["fractal_power"]

    state.current_fractal = FractalType.JULIA
    state.needs_update = True


def apply_mandelbrot_preset(state, preset, index):

    state.mandelbrot_preset_index = index
    state.mandelbrot_preset_name = preset["name"]

    state.mandelbrot_zx = preset["mandelbrot_zx"]
    state.mandelbrot_zy = preset["mandelbrot_zy"]

    if preset.get("fractal_power") in POWER_VALUES:
        state.fractal_power = preset["fractal_power"]

    state.current_fractal = FractalType.MANDELBROT
    state.needs_update = True


def begin_preset_name_input(state, target):

    state.preset_name_input_active = True
    state.preset_name_input = ""
    state.preset_name_target = target


def cancel_preset_name_input(state):

    state.preset_name_input_active = False
    state.preset_name_input = ""
    state.preset_name_target = None


def save_current_custom_preset(state):

    name = state.preset_name_input.strip()

    if not name:
        cancel_preset_name_input(state)
        return

    if state.preset_name_target == "julia":
        state.custom_presets["julia"].append({
            "name": name,
            "julia_cx": state.julia_cx,
            "julia_cy": state.julia_cy,
            "fractal_power": state.fractal_power,
        })

        state.julia_preset_index = len(
            get_combined_julia_presets(state)
        ) - 1

        state.julia_preset_name = name

    elif state.preset_name_target == "mandelbrot":
        state.custom_presets["mandelbrot"].append({
            "name": name,
            "mandelbrot_zx": state.mandelbrot_zx,
            "mandelbrot_zy": state.mandelbrot_zy,
            "fractal_power": state.fractal_power,
        })

        state.mandelbrot_preset_index = len(
            get_combined_mandelbrot_presets(state)
        ) - 1

        state.mandelbrot_preset_name = name

    save_user_presets(state.custom_presets)
    cancel_preset_name_input(state)


def delete_custom_preset(state, target, combined_index):

    builtin_count = (
        len(JULIA_PRESETS)
        if target == "julia"
        else len(MANDELBROT_PRESETS)
    )

    custom_index = combined_index - builtin_count

    if custom_index < 0:
        return

    if custom_index >= len(state.custom_presets[target]):
        return

    del state.custom_presets[target][custom_index]

    if target == "julia":
        state.julia_preset_index = 0
        state.julia_preset_name = JULIA_PRESETS[0][0]

    else:
        state.mandelbrot_preset_index = 0
        state.mandelbrot_preset_name = MANDELBROT_PRESETS[0][0]

    save_user_presets(state.custom_presets)


# ============================================================================
# MENUS
# ============================================================================

BASE_MENU_OPTIONS = [
    "Continue",
    "Fractal Settings",
    "Visual Settings",
    "Export Settings",
    "Audio Settings",
    "Controls",
    "Tutorial",
    "Achievements",
    "Exit"
]


def get_fractal_settings_options(state):

    options = [
        "Fractal Type",
        "Iterations",
    ]

    if state.current_fractal in (
        FractalType.MANDELBROT,
        FractalType.JULIA
    ):
        options.append("Power")

    if state.current_fractal == FractalType.MANDELBROT:
        options.append("Mandelbrot Presets")

    if state.current_fractal == FractalType.JULIA:
        options.append("Julia Presets")

    return options


def get_visual_settings_options(state):

    return [
        "Palette",
        "Game Resolution",
        "Show Performance Info",
        "Benchmark GPU vs CPU",
    ]


def get_export_settings_options(state):

    return [
        "PNG Resolution",
        "GIF Resolution",
        "Export PNG",
        "Export GIF",
    ]


def get_audio_settings_options(state):

    return []


def get_category_options(state, category):

    if category == "Fractal Settings":
        return get_fractal_settings_options(state)

    if category == "Visual Settings":
        return get_visual_settings_options(state)

    if category == "Export Settings":
        return get_export_settings_options(state)

    if category == "Audio Settings":
        return get_audio_settings_options(state)

    return []


def get_menu_options(state):
    """ Builds the active main menu option list.
    Conditional fractal options live inside the Fractal Settings
    submenu so the top-level settings menu stays short.
    """

    return BASE_MENU_OPTIONS.copy()

# ============================================================================
# APP STATE
# ============================================================================


@dataclass
class AppState:
    """ Stores all mutable runtime application state.
    The application uses a centralized state container so
    that the renderer, UI system, menu system, camera,
    export system, and input handler can operate on a
    shared synchronized runtime model.
    """ 


    # --------------------------------------------------------
    # Rendering State
    # --------------------------------------------------------
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    max_iter: int

    current_palette: int

    current_fractal: FractalType

    fractal_power: int = 2

    running: bool = True

    # Indicates whether the fractal surface must be re-rendered.
    needs_update: bool = True

    show_ui: bool = True

    menu_open: bool = False
    submenu_open: str | None = None
    menu_index: int = 0
    hovered_menu_item: str | None = None

    user_progress: dict = field(
        default_factory=default_user_progress
    )

    tutorial_active: bool = False
    tutorial_step_index: int = 0

    custom_presets: dict = field(
        default_factory=empty_user_presets
    )

    preset_name_input_active: bool = False
    preset_name_input: str = ""
    preset_name_target: str | None = None

    surface: pg.Surface | None = None

    # --------------------------------------------------------
    # Export State
    # --------------------------------------------------------
    exporting: bool = False
    export_message: str = ""
    export_message_timer: float = 0.0

    benchmark_results_open: bool = False
    benchmark_result_lines: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Render Performance Statistics
    # --------------------------------------------------------
    last_render_time_ms: float = 0.0
    last_render_fps: float = 0.0

    # --------------------------------------------------------
    # Mandelbrot Fractal Configuration
    # --------------------------------------------------------
    mandelbrot_zx: float = 0.0
    mandelbrot_zy: float = 0.0

    mandelbrot_preset_index: int = 0
    mandelbrot_preset_name: str = "Classic"

    # --------------------------------------------------------
    # Julia Fractal Configuration
    # --------------------------------------------------------
    julia_cx: float = -0.8
    julia_cy: float = 0.156

    julia_preset_index: int = 0
    julia_preset_name: str = "Classic"



# ============================================================================
# INITIAL STATE
# ============================================================================


def create_initial_state():


    """ Creates the default runtime application state.
    The initial viewport is configured to frame the
    Mandelbrot Set while preserving the display aspect ratio.
    
    Returns:
        Initialized AppState instance.
    """

    xmin, xmax = -3.0, 1.5

    x_range = xmax - xmin

    # Preserve aspect ratio when constructing the viewport.
    y_range = x_range * (
        CONFIG.height / CONFIG.width
    )

    ymin = -y_range / 2
    ymax = y_range / 2

    return AppState(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        max_iter=100,
        current_palette=0,
        current_fractal=FractalType.MANDELBROT,
    )


def is_parameter_edit_locked(state):

    return abs(state.xmax - state.xmin) < PARAMETER_EDIT_LOCK_WIDTH


def show_parameter_lock_message(state):

    state.export_message = (
        "Parameter changes locked at deep zoom"
    )

    state.export_message_timer = time.time()

# ============================================================================
# CUDA KERNEL
# ============================================================================

""" The CUDA kernel performs GPU-accelerated escape-time fractal rendering using double-double precision arithmetic.

Rendering stages:

Convert screen coordinates into viewport coordinates.
Initialize fractal state.
Execute iterative fractal calculations.
Perform escape-radius checks.
Store iteration counts into the GPU output buffer.

Double-double arithmetic is used to improve numerical stability during deep zoom operations. """
fractal_kernel = cp.RawKernel(r'''

struct dd
{
    double hi;
    double lo;
};

__device__ inline dd dd_set(double a)
{
    dd r;
    r.hi = a;
    r.lo = 0.0;
    return r;
}

__device__ inline dd quick_two_sum(double a, double b)
{
    dd r;

    r.hi = a + b;
    r.lo = b - (r.hi - a);

    return r;
}

__device__ inline dd two_sum(double a, double b)
{
    dd r;

    r.hi = a + b;

    double bb = r.hi - a;

    r.lo =
        (a - (r.hi - bb)) +
        (b - bb);

    return r;
}

__device__ inline dd split(double a)
{
    const double splitter = 134217729.0;

    double t = splitter * a;

    dd r;

    r.hi = t - (t - a);
    r.lo = a - r.hi;

    return r;
}

__device__ inline dd two_prod(double a, double b)
{
    dd r;

    r.hi = a * b;

    dd sa = split(a);
    dd sb = split(b);

    r.lo =
        ((sa.hi * sb.hi - r.hi) +
        sa.hi * sb.lo +
        sa.lo * sb.hi) +
        sa.lo * sb.lo;

    return r;
}

__device__ inline dd dd_add(dd a, dd b)
{
    dd s = two_sum(a.hi, b.hi);

    double e =
        a.lo +
        b.lo +
        s.lo;

    return quick_two_sum(s.hi, e);
}

__device__ inline dd dd_sub(dd a, dd b)
{
    dd s = two_sum(a.hi, -b.hi);

    double e =
        a.lo -
        b.lo +
        s.lo;

    return quick_two_sum(s.hi, e);
}

__device__ inline dd dd_mul(dd a, dd b)
{
    dd p = two_prod(a.hi, b.hi);

    p.lo +=
        a.hi * b.lo +
        a.lo * b.hi;

    return quick_two_sum(p.hi, p.lo);
}

__device__ inline dd dd_abs(dd a)
{
    if (a.hi < 0.0)
    {
        a.hi = -a.hi;
        a.lo = -a.lo;
    }

    return a;
}

__device__ inline double dd_to_double(dd a)
{
    return a.hi + a.lo;
}

__device__ inline void complex_mul(
    dd ar,
    dd ai,
    dd br,
    dd bi,
    dd* rr,
    dd* ri
)
{
    dd arbr = dd_mul(ar, br);
    dd aibi = dd_mul(ai, bi);

    dd arbi = dd_mul(ar, bi);
    dd aibr = dd_mul(ai, br);

    *rr = dd_sub(arbr, aibi);
    *ri = dd_add(arbi, aibr);
}

extern "C" __global__
void fractal(
    int fractal_type,

    double xmin,
    double xmax,
    double ymin,
    double ymax,

    int width,
    int height,

    int max_iter,

    double julia_cx,
    double julia_cy,

    double mandelbrot_zx,
    double mandelbrot_zy,

    int fractal_power,

    int* iter_out
)
{
    int x =
        blockIdx.x *
        blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y *
        blockDim.y +
        threadIdx.y;

    if (x >= width || y >= height)
        return;

    // ============================================================
    // DOUBLE-DOUBLE VIEWPORT SETUP
    // ============================================================

    dd dd_xmin = dd_set(xmin);
    dd dd_xmax = dd_set(xmax);

    dd dd_ymin = dd_set(ymin);
    dd dd_ymax = dd_set(ymax);

    dd dx = dd_mul(
        dd_sub(dd_xmax, dd_xmin),
        dd_set(1.0 / (double)width)
    );

    dd dy = dd_mul(
        dd_sub(dd_ymax, dd_ymin),
        dd_set(1.0 / (double)height)
    );

    dd cr = dd_add(
        dd_xmin,
        dd_mul(
            dd_set((double)x),
            dx
        )
    );

    dd ci = dd_sub(
        dd_ymax,
        dd_mul(
            dd_set((double)y),
            dy
        )
    );

    // ============================================================
    // INITIAL VALUES
    // ============================================================

    dd z0r;
    dd z0i;

    dd c_r;
    dd c_i;

    if (fractal_type == 0)
    {
        // MANDELBROT

        z0r = dd_set(mandelbrot_zx);
        z0i = dd_set(mandelbrot_zy);

        c_r = cr;
        c_i = ci;
    }
    else if (fractal_type == 1)
    {
        // JULIA

        z0r = cr;
        z0i = ci;

        c_r = dd_set(julia_cx);
        c_i = dd_set(julia_cy);
    }
    else
    {
        // BURNING SHIP

        z0r = dd_set(0.0);
        z0i = dd_set(0.0);

        c_r = cr;
        c_i = ci;
    }

    dd zr = z0r;
    dd zi = z0i;

    int i;

    for (i = 0; i < max_iter; i++)
    {
        if (fractal_type == 2)
        {
            zr = dd_abs(zr);
            zi = dd_abs(zi);
        }

        dd zr2 = dd_mul(zr, zr);
        dd zi2 = dd_mul(zi, zi);

        dd mag2 = dd_add(zr2, zi2);

        if (mag2.hi > 4.0)
            break;

        dd zrzi = dd_mul(zr, zi);

        dd two_zrzi = dd_add(
            zrzi,
            zrzi
        );

        dd z2r = dd_sub(zr2, zi2);
        dd z2i = two_zrzi;

        dd power_r = z2r;
        dd power_i = z2i;

        if (fractal_type != 2 && fractal_power > 2)
        {
            dd z3r;
            dd z3i;

            complex_mul(
                z2r,
                z2i,
                zr,
                zi,
                &z3r,
                &z3i
            );

            if (fractal_power == 3)
            {
                power_r = z3r;
                power_i = z3i;
            }
            else
            {
                dd z4r;
                dd z4i;

                complex_mul(
                    z2r,
                    z2i,
                    z2r,
                    z2i,
                    &z4r,
                    &z4i
                );

                if (fractal_power == 4)
                {
                    power_r = z4r;
                    power_i = z4i;
                }
                else if (fractal_power == 5)
                {
                    complex_mul(
                        z4r,
                        z4i,
                        zr,
                        zi,
                        &power_r,
                        &power_i
                    );
                }
                else
                {
                    complex_mul(
                        z3r,
                        z3i,
                        z3r,
                        z3i,
                        &power_r,
                        &power_i
                    );
                }
            }
        }

        dd zr_new = dd_add(
            power_r,
            c_r
        );

        dd zi_new = dd_add(
            power_i,
            c_i
        );

        zr = zr_new;
        zi = zi_new;
    }

    

    int idx =
        y * width + x;

    iter_out[idx] = i;
}

''', 'fractal')

# ============================================================================
# PALETTES
# ============================================================================


def smooth_gradient(stops, t):

    t = cp.clip(t, 0.0, 1.0)

    result_r = cp.zeros_like(t)
    result_g = cp.zeros_like(t)
    result_b = cp.zeros_like(t)

    for i in range(len(stops) - 1):

        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]

        mask = (t >= p0) & (t <= p1)

        local_t = (t - p0) / max(p1 - p0, 1e-8)

        local_t = local_t * local_t * (
            3.0 - 2.0 * local_t
        )

        result_r = cp.where(
            mask,
            c0[0] + (c1[0] - c0[0]) * local_t,
            result_r
        )

        result_g = cp.where(
            mask,
            c0[1] + (c1[1] - c0[1]) * local_t,
            result_g
        )

        result_b = cp.where(
            mask,
            c0[2] + (c1[2] - c0[2]) * local_t,
            result_b
        )

    return result_r, result_g, result_b


def gradient_palette(name, stops):

    return {
        "name": name,
        "func": lambda t: smooth_gradient(stops, t)
    }


PALETTES = [

    gradient_palette(
        "Classic Fire",
        [
            (0.0, (0, 0, 0)),
            (0.3, (255, 0, 0)),
            (0.6, (255, 180, 0)),
            (1.0, (255, 255, 255)),
        ]
    ),

    gradient_palette(
        "Ocean",
        [
            (0.0, (0, 7, 100)),
            (0.3, (32, 107, 203)),
            (0.6, (0, 200, 255)),
            (1.0, (220, 255, 255)),
        ]
    ),

    gradient_palette(
        "Inferno",
        [
            (0.0, (0, 0, 0)),
            (0.3, (100, 0, 50)),
            (0.6, (255, 100, 0)),
            (1.0, (255, 255, 200)),
        ]
    ),

    gradient_palette(
        "Rainbow",
        [
            (0.0, (255, 0, 0)),
            (0.2, (255, 127, 0)),
            (0.4, (255, 255, 0)),
            (0.6, (0, 255, 0)),
            (0.8, (0, 0, 255)),
            (1.0, (148, 0, 211)),
        ]
    ),

    gradient_palette(
        "Neon",
        [
            (0.0, (0, 0, 0)),
            (0.3, (255, 0, 255)),
            (0.6, (0, 255, 255)),
            (1.0, (255, 255, 255)),
        ]
    ),

    gradient_palette(
        "Icy",
        [
            (0.0, (0, 0, 20)),
            (0.3, (0, 120, 255)),
            (0.7, (180, 240, 255)),
            (1.0, (255, 255, 255)),
        ]
    ),

    gradient_palette(
        "Sunset",
        [
            (0.0, (0, 0, 0)),
            (0.2, (80, 0, 120)),
            (0.5, (255, 60, 60)),
            (0.8, (255, 180, 0)),
            (1.0, (255, 255, 200)),
        ]
    ),

    gradient_palette(
        "B&W High Contrast",
        [
            (0.0, (0, 0, 0)),
            (0.49, (0, 0, 0)),
            (0.5, (255, 255, 255)),
            (1.0, (255, 255, 255)),
        ]
    ),

]

# ============================================================================
# COLORIZE
# ============================================================================


def colorize_gpu(iter_arr, max_iter, palette_index, width, height):

    t = iter_arr.astype(cp.float32) / max_iter

    palette = PALETTES[palette_index]["func"]

    r, g, b = palette(t)

    inside = iter_arr >= max_iter

    r = cp.where(inside, 0, r)
    g = cp.where(inside, 0, g)
    b = cp.where(inside, 0, b)

    r = cp.clip(r, 0, 255).astype(cp.uint8)
    g = cp.clip(g, 0, 255).astype(cp.uint8)
    b = cp.clip(b, 0, 255).astype(cp.uint8)

    rgb_flat = cp.stack([r, g, b], axis=1)

    return rgb_flat.reshape(height, width, 3)

# ============================================================================
# FRACTAL RENDER
# ============================================================================


def render_fractal_iterations_gpu(
    state,
    width,
    height
):

    gpu_buffer = cp.zeros(
        width * height,
        dtype=cp.int32
    )

    block = (16, 16)

    grid = (
        (width + block[0] - 1) // block[0],
        (height + block[1] - 1) // block[1],
    )

    fractal_type = 0

    if state.current_fractal == FractalType.JULIA:
        fractal_type = 1

    elif state.current_fractal == FractalType.BURNING_SHIP:
        fractal_type = 2

    fractal_kernel(
        grid,
        block,
        (
            np.int32(fractal_type),

            np.float64(state.xmin),
            np.float64(state.xmax),
            np.float64(state.ymin),
            np.float64(state.ymax),

            np.int32(width),
            np.int32(height),

            np.int32(state.max_iter),

            np.float64(state.julia_cx),
            np.float64(state.julia_cy),

            np.float64(state.mandelbrot_zx),
            np.float64(state.mandelbrot_zy),

            np.int32(state.fractal_power),

            gpu_buffer,
        )
    )

    return gpu_buffer


def render_fractal_gpu(
    state,
    width,
    height
):

    gpu_buffer = render_fractal_iterations_gpu(
        state,
        width,
        height
    )

    rgb_hw3 = colorize_gpu(
        gpu_buffer,
        state.max_iter,
        state.current_palette,
        width,
        height
    )

    return rgb_hw3


def render_fractal_iterations_cpu(
    state,
    width,
    height
):

    x = np.arange(
        width,
        dtype=np.float64
    )

    y = np.arange(
        height,
        dtype=np.float64
    )

    cr = state.xmin + (
        x *
        ((state.xmax - state.xmin) / width)
    )

    ci = state.ymax - (
        y *
        ((state.ymax - state.ymin) / height)
    )

    cr, ci = np.meshgrid(
        cr,
        ci
    )

    if state.current_fractal == FractalType.MANDELBROT:

        zr = np.full(
            (height, width),
            state.mandelbrot_zx,
            dtype=np.float64
        )

        zi = np.full(
            (height, width),
            state.mandelbrot_zy,
            dtype=np.float64
        )

        c_r = cr
        c_i = ci

    elif state.current_fractal == FractalType.JULIA:

        zr = cr.copy()
        zi = ci.copy()

        c_r = np.full(
            (height, width),
            state.julia_cx,
            dtype=np.float64
        )

        c_i = np.full(
            (height, width),
            state.julia_cy,
            dtype=np.float64
        )

    else:

        zr = np.zeros(
            (height, width),
            dtype=np.float64
        )

        zi = np.zeros(
            (height, width),
            dtype=np.float64
        )

        c_r = cr
        c_i = ci

    iter_out = np.full(
        (height, width),
        state.max_iter,
        dtype=np.int32
    )

    active = np.ones(
        (height, width),
        dtype=bool
    )

    with np.errstate(
        over="ignore",
        invalid="ignore"
    ):

        for i in range(state.max_iter):

            if state.current_fractal == FractalType.BURNING_SHIP:
                zr = np.abs(zr)
                zi = np.abs(zi)

            mag2 = zr * zr + zi * zi

            escaped = active & (mag2 > 4.0)

            iter_out[escaped] = i
            active &= ~escaped

            if not active.any():
                break

            if state.current_fractal != FractalType.BURNING_SHIP:

                if state.fractal_power == 2:
                    zr_active = zr[active]
                    zi_active = zi[active]

                    power_r = (
                        zr_active *
                        zr_active -
                        zi_active *
                        zi_active
                    )

                    power_i = (
                        2.0 *
                        zr_active *
                        zi_active
                    )

                else:
                    z = (
                        zr[active] +
                        1j *
                        zi[active]
                    ) ** state.fractal_power

                    power_r = z.real
                    power_i = z.imag

            else:
                zr_active = zr[active]
                zi_active = zi[active]

                power_r = (
                    zr_active *
                    zr_active -
                    zi_active *
                    zi_active
                )

                power_i = (
                    2.0 *
                    zr_active *
                    zi_active
                )

            zr[active] = (
                power_r +
                c_r[active]
            )

            zi[active] = (
                power_i +
                c_i[active]
            )

    return iter_out


def run_gpu_cpu_benchmark(state):

    try:

        state.exporting = True

        state.export_message = (
            "Running benchmark..."
        )

        state.export_message_timer = time.time()

        render_fractal_iterations_gpu(
            state,
            BENCHMARK_WIDTH,
            BENCHMARK_HEIGHT
        )

        cp.cuda.Stream.null.synchronize()

        gpu_times = []

        for _ in range(BENCHMARK_RUNS):

            start = time.perf_counter()

            render_fractal_iterations_gpu(
                state,
                BENCHMARK_WIDTH,
                BENCHMARK_HEIGHT
            )

            cp.cuda.Stream.null.synchronize()

            gpu_times.append(
                (time.perf_counter() - start) * 1000.0
            )

        cpu_times = []

        for _ in range(BENCHMARK_RUNS):

            start = time.perf_counter()

            render_fractal_iterations_cpu(
                state,
                BENCHMARK_WIDTH,
                BENCHMARK_HEIGHT
            )

            cpu_times.append(
                (time.perf_counter() - start) * 1000.0
            )

        gpu_avg = sum(gpu_times) / len(gpu_times)
        cpu_avg = sum(cpu_times) / len(cpu_times)

        if gpu_avg > 0.0:
            speedup = cpu_avg / gpu_avg
        else:
            speedup = 0.0

        state.benchmark_result_lines = [
            "GPU vs CPU Benchmark",
            "",
            f"Resolution: {BENCHMARK_WIDTH}x{BENCHMARK_HEIGHT}",
            f"Runs: {BENCHMARK_RUNS}",
            f"Fractal: {state.current_fractal.value}",
            f"Iterations: {state.max_iter}",
            f"GPU average: {gpu_avg:.1f} ms",
            f"CPU average: {cpu_avg:.1f} ms",
            f"GPU speedup: {speedup:.1f}x",
            "",
            "Press ESC to close",
        ]

        state.benchmark_results_open = True

        state.export_message = (
            "Benchmark complete"
        )

        state.export_message_timer = time.time()

        unlock_achievement(state, "benchmark_runner")

    except Exception as e:

        state.benchmark_result_lines = [
            "GPU vs CPU Benchmark",
            "",
            f"BENCHMARK ERROR: {e}",
            "",
            "Press ESC to close",
        ]

        state.benchmark_results_open = True

        state.export_message = (
            f"BENCHMARK ERROR: {e}"
        )

        state.export_message_timer = time.time()

    finally:

        state.exporting = False


def compute_surface(state, width, height):

    rgb_hw3 = render_fractal_gpu(
        state,
        width,
        height
    )

    rgb_wh3 = cp.transpose(
        rgb_hw3,
        (1, 0, 2)
    )

    return pg.surfarray.make_surface(
        cp.asnumpy(rgb_wh3)
    )

# ============================================================================
# EXPORT
# ============================================================================


def ensure_export_folders():

    os.makedirs(
        PNG_EXPORT_DIR,
        exist_ok=True
    )

    os.makedirs(
        GIF_EXPORT_DIR,
        exist_ok=True
    )


def export_fractal_png(state):

    try:

        state.exporting = True

        state.export_message = (
            "Rendering export..."
        )

        rgb_hw3 = render_fractal_gpu(
            state,
            CONFIG.png_export_width,
            CONFIG.png_export_height
        )

        rgb_wh3 = cp.transpose(
            rgb_hw3,
            (1, 0, 2)
        )

        surface = pg.surfarray.make_surface(
            cp.asnumpy(rgb_wh3)
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        ensure_export_folders()

        filename = (
            f"fractal_{timestamp}.png"
        )

        filepath = os.path.join(
            PNG_EXPORT_DIR,
            filename
        )

        pg.image.save(surface, filepath)

        state.exporting = False

        state.export_message = (
            f"Exported: {filepath}"
        )

        state.export_message_timer = time.time()

        unlock_achievement(state, "first_export")

    except Exception as e:

        state.exporting = False

        state.export_message = (
            f"EXPORT ERROR: {e}"
        )

        state.export_message_timer = time.time()



def get_gif_export_size():

    width = min(
        CONFIG.gif_export_width,
        GIF_MAX_WIDTH
    )

    height = min(
        CONFIG.gif_export_height,
        GIF_MAX_HEIGHT
    )

    width = max(256, int(width))
    height = max(256, int(height))

    return width, height


def export_zoom_gif(state):

    if imageio is None:

        state.export_message = (
            "GIF export requires imageio"
        )

        state.export_message_timer = time.time()
        return

    saved_viewport = (
        state.xmin,
        state.xmax,
        state.ymin,
        state.ymax,
    )

    try:

        state.exporting = True

        state.export_message = (
            "Rendering GIF..."
        )

        state.export_message_timer = time.time()

        gif_width, gif_height = get_gif_export_size()

        target_center_x = (
            state.xmin +
            state.xmax
        ) * 0.5

        target_center_y = (
            state.ymin +
            state.ymax
        ) * 0.5

        target_range_x = (
            state.xmax -
            state.xmin
        )

        target_range_y = (
            state.ymax -
            state.ymin
        )

        default_state = create_initial_state()

        start_range_x = (
            default_state.xmax -
            default_state.xmin
        )

        start_range_y = (
            default_state.ymax -
            default_state.ymin
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        ensure_export_folders()

        filename = (
            f"fractal_zoom_{timestamp}.gif"
        )

        filepath = os.path.join(
            GIF_EXPORT_DIR,
            filename
        )

        with imageio.get_writer(
            filepath,
            mode="I",
            duration=GIF_DURATION_MS / 1000.0
        ) as writer:

            for frame_index in range(GIF_FRAME_COUNT):

                if GIF_FRAME_COUNT <= 1:
                    t = 1.0
                else:
                    t = frame_index / (
                        GIF_FRAME_COUNT - 1
                    )

                current_range_x = (
                    start_range_x *
                    (
                        target_range_x /
                        start_range_x
                    ) ** t
                )

                current_range_y = (
                    start_range_y *
                    (
                        target_range_y /
                        start_range_y
                    ) ** t
                )

                state.xmin = target_center_x - current_range_x * 0.5
                state.xmax = target_center_x + current_range_x * 0.5
                state.ymin = target_center_y - current_range_y * 0.5
                state.ymax = target_center_y + current_range_y * 0.5

                rgb_hw3 = render_fractal_gpu(
                    state,
                    gif_width,
                    gif_height
                )

                writer.append_data(
                    cp.asnumpy(rgb_hw3)
                )

        state.export_message = (
            f"Exported GIF: {filepath}"
        )

        state.export_message_timer = time.time()

        unlock_achievement(state, "first_export")

    except Exception as e:

        state.export_message = (
            f"GIF EXPORT ERROR: {e}"
        )

        state.export_message_timer = time.time()

    finally:

        (
            state.xmin,
            state.xmax,
            state.ymin,
            state.ymax,
        ) = saved_viewport

        state.exporting = False
        state.needs_update = True

# ============================================================================
# CAMERA
# ============================================================================


class Camera:

    @staticmethod
    def get_move_speed(state):

        return (
            CONFIG.base_move_speed *
            (state.xmax - state.xmin)
        )

    @staticmethod
    def move(state, dx, dy):

        speed = Camera.get_move_speed(state)

        state.xmin += dx * speed
        state.xmax += dx * speed

        state.ymin += dy * speed
        state.ymax += dy * speed

        state.needs_update = True
        check_deep_zoom_achievement(state)

    @staticmethod
    def center_on_pixel(state, mx, my):

        cx = (
            state.xmin +
            (mx / CONFIG.width) *
            (state.xmax - state.xmin)
        )

        cy = (
            state.ymax -
            (my / CONFIG.height) *
            (state.ymax - state.ymin)
        )

        xr = state.xmax - state.xmin
        yr = state.ymax - state.ymin

        state.xmin = cx - xr / 2
        state.xmax = cx + xr / 2

        state.ymin = cy - yr / 2
        state.ymax = cy + yr / 2

        state.needs_update = True
        check_deep_zoom_achievement(state)

    @staticmethod
    def zoom(state, direction):

        cx = (
            state.xmin +
            state.xmax
        ) * 0.5

        cy = (
            state.ymin +
            state.ymax
        ) * 0.5

        xr = (
            state.xmax -
            state.xmin
        )

        yr = (
            state.ymax -
            state.ymin
        )

        zoom = 0.8 if direction > 0 else 1.25

        xr *= zoom
        yr *= zoom

        # ============================================================
        # FLOATING POINT PRECISION LIMIT
        #
        # Prevents xmin/xmax and ymin/ymax from collapsing into the
        # same value due to IEEE double precision limits.
        # ============================================================

        EPS = 1e-15

        if abs(xr) < EPS * max(1.0, abs(cx)):
            return

        if abs(yr) < EPS * max(1.0, abs(cy)):
            return

        state.xmin = cx - xr * 0.5
        state.xmax = cx + xr * 0.5

        state.ymin = cy - yr * 0.5
        state.ymax = cy + yr * 0.5

        state.needs_update = True
        check_deep_zoom_achievement(state)

# ============================================================================
# RUNTIME SETTINGS HELPERS
# ============================================================================


def preserve_viewport_for_aspect_ratio(state):

    center_x = (
        state.xmin +
        state.xmax
    ) * 0.5

    center_y = (
        state.ymin +
        state.ymax
    ) * 0.5

    x_range = (
        state.xmax -
        state.xmin
    )

    y_range = x_range * (
        CONFIG.height /
        CONFIG.width
    )

    state.xmin = center_x - x_range * 0.5
    state.xmax = center_x + x_range * 0.5
    state.ymin = center_y - y_range * 0.5
    state.ymax = center_y + y_range * 0.5


def apply_game_resolution(state, width, height):

    CONFIG.upscale = 1
    CONFIG.base_width = int(width)
    CONFIG.base_height = int(height)

    preserve_viewport_for_aspect_ratio(state)

    pg.display.set_mode(
        (
            CONFIG.width,
            CONFIG.height
        )
    )

    save_current_user_settings()

    state.surface = None
    state.needs_update = True


def set_png_resolution(width, height):

    CONFIG.png_export_width = int(width)
    CONFIG.png_export_height = int(height)

    save_current_user_settings()


def set_gif_resolution(width, height):

    CONFIG.gif_export_width = int(width)
    CONFIG.gif_export_height = int(height)

    save_current_user_settings()


def set_volume(audio_manager, key, value):

    if key == "music_volume":
        audio_manager.set_music_volume(value)

        update_and_save_user_setting(
            "music_volume",
            CONFIG.music_volume
        )

    elif key == "ui_volume":
        audio_manager.set_ui_volume(value)

        update_and_save_user_setting(
            "ui_volume",
            CONFIG.ui_volume
        )


def change_volume(audio_manager, key, delta):

    if key == "music_volume":
        set_volume(
            audio_manager,
            key,
            CONFIG.music_volume + delta
        )

    elif key == "ui_volume":
        set_volume(
            audio_manager,
            key,
            CONFIG.ui_volume + delta
        )


def toggle_performance_info():

    CONFIG.show_performance_info = (
        not CONFIG.show_performance_info
    )

    update_and_save_user_setting(
        "show_performance_info",
        CONFIG.show_performance_info
    )

# ============================================================================
# INPUT
# ============================================================================


class InputHandler:

    @staticmethod
    def process_events(state, audio_manager):

        for event in pg.event.get():

            if event.type == pg.QUIT:
                state.running = False

            elif state.tutorial_active:
                InputHandler.handle_tutorial_event(
                    state,
                    event,
                    audio_manager
                )

            elif event.type == pg.KEYDOWN:
                InputHandler.handle_keydown(
                    state,
                    event,
                    audio_manager
                )

            elif event.type == pg.MOUSEBUTTONDOWN:
                InputHandler.handle_mouse(
                    state,
                    event,
                    audio_manager
                )

    @staticmethod
    def handle_tutorial_event(state, event, audio_manager):

        if event.type == pg.KEYDOWN:

            if event.key == pg.K_ESCAPE:

                if get_tutorial_action(state) == "menu":
                    handle_tutorial_action(
                        state,
                        "menu",
                        audio_manager
                    )

                else:
                    skip_tutorial(state)

                    if audio_manager:
                        audio_manager.play_click()

                return

            if event.key == pg.K_RETURN:

                action = get_tutorial_action(state)

                if action == "begin":
                    handle_tutorial_action(
                        state,
                        "begin",
                        audio_manager
                    )

                elif action == "finish":
                    handle_tutorial_action(
                        state,
                        "finish",
                        audio_manager
                    )

                elif action == "parameters" and not is_tutorial_parameter_step_available(state):
                    handle_tutorial_action(
                        state,
                        "parameter_continue",
                        audio_manager
                    )

                return

            if event.key in (
                pg.K_w,
                pg.K_a,
                pg.K_s,
                pg.K_d
            ):
                InputHandler.handle_keydown(
                    state,
                    event,
                    audio_manager
                )

                handle_tutorial_action(
                    state,
                    "movement",
                    audio_manager
                )
                return

            if event.key in (pg.K_q, pg.K_e):
                InputHandler.handle_keydown(
                    state,
                    event,
                    audio_manager
                )

                handle_tutorial_action(
                    state,
                    "iterations",
                    audio_manager
                )
                return

            if event.key in (pg.K_r, pg.K_f):
                InputHandler.handle_keydown(
                    state,
                    event,
                    audio_manager
                )

                handle_tutorial_action(
                    state,
                    "palette",
                    audio_manager
                )
                return

            if event.key in (
                pg.K_i,
                pg.K_j,
                pg.K_k,
                pg.K_l
            ):
                InputHandler.handle_keydown(
                    state,
                    event,
                    audio_manager
                )

                if is_tutorial_parameter_step_available(state):
                    handle_tutorial_action(
                        state,
                        "parameters",
                        audio_manager
                    )

                return

        elif event.type == pg.MOUSEBUTTONDOWN:

            if event.button == 1:
                InputHandler.handle_mouse(
                    state,
                    event,
                    audio_manager
                )

                handle_tutorial_action(
                    state,
                    "center",
                    audio_manager
                )
                return

            if event.button in (4, 5):
                InputHandler.handle_mouse(
                    state,
                    event,
                    audio_manager
                )

                handle_tutorial_action(
                    state,
                    "zoom",
                    audio_manager
                )
                return

    @staticmethod
    def activate_menu_option(state, option, audio_manager=None):

        if audio_manager:
            audio_manager.play_click()

        if option == "Continue":

            state.menu_open = False
            state.submenu_open = None

        elif option == "Export PNG":

            export_fractal_png(state)

        elif option == "Export GIF":

            export_zoom_gif(state)

        elif option == "Show Performance Info":

            toggle_performance_info()

        elif option == "Benchmark GPU vs CPU":

            run_gpu_cpu_benchmark(state)

        elif option == "Tutorial":

            state.menu_open = False
            state.submenu_open = None
            state.tutorial_step_index = 0
            state.tutorial_active = True

        elif option == "Exit":

            state.running = False

        else:

            state.submenu_open = option

    @staticmethod
    def handle_keydown(state, event, audio_manager):

        if state.preset_name_input_active:

            if event.key == pg.K_ESCAPE:
                cancel_preset_name_input(state)
                return

            if event.key == pg.K_RETURN:
                save_current_custom_preset(state)

                if audio_manager:
                    audio_manager.play_click()

                return

            if event.key == pg.K_BACKSPACE:
                state.preset_name_input = (
                    state.preset_name_input[:-1]
                )
                return

            if event.unicode and event.unicode.isprintable():
                if len(state.preset_name_input) < 32:
                    state.preset_name_input += event.unicode

            return

        if state.benchmark_results_open:

            if event.key == pg.K_ESCAPE:
                state.benchmark_results_open = False

            return

        if event.key == pg.K_ESCAPE:

            if state.submenu_open:

                state.submenu_open = None

            else:

                state.menu_open = (
                    not state.menu_open
                )

            return

        # ========================================================
        # MENU SHORTCUTS
        # ========================================================

        if state.submenu_open == "Iterations":

            if event.key == pg.K_q:
                state.max_iter = max(
                    1,
                    state.max_iter // 2
                )

                state.needs_update = True

            elif event.key == pg.K_e:
                state.max_iter *= 2
                state.needs_update = True

        if state.submenu_open == "Show Performance Info":

            if event.key in (pg.K_RETURN, pg.K_LEFT, pg.K_RIGHT):
                toggle_performance_info()

        if state.menu_open:

            if event.key == pg.K_DOWN:

                state.menu_index += 1
                state.menu_index %= len(
                    get_menu_options(state)
                )

            elif event.key == pg.K_RETURN:

                option = get_menu_options(state)[
                    state.menu_index
                ]

                InputHandler.activate_menu_option(
                    state,
                    option,
                    audio_manager
                )

            return

        # ========================================================
        # NORMAL CONTROLS
        # ========================================================

        if event.key == pg.K_w:
            Camera.move(state, 0, 1)

        elif event.key == pg.K_s:
            Camera.move(state, 0, -1)

        elif event.key == pg.K_a:
            Camera.move(state, -1, 0)

        elif event.key == pg.K_d:
            Camera.move(state, 1, 0)

        elif event.key == pg.K_q:

            state.max_iter = max(
                1,
                state.max_iter // 2
            )

            state.needs_update = True

        elif event.key == pg.K_e:

            state.max_iter *= 2
            state.needs_update = True

        elif event.key == pg.K_r:

            state.current_palette += 1
            state.current_palette %= len(
                PALETTES
            )

            state.needs_update = True
            unlock_achievement(state, "palette_swapper")

        elif event.key == pg.K_f:

            state.current_palette -= 1
            state.current_palette %= len(
                PALETTES
            )

            state.needs_update = True
            unlock_achievement(state, "palette_swapper")

        elif event.key == pg.K_p:

            export_fractal_png(state)

        elif event.key in (
            pg.K_i,
            pg.K_k,
            pg.K_j,
            pg.K_l
        ):

            if state.current_fractal in (
                FractalType.MANDELBROT,
                FractalType.JULIA
            ) and is_parameter_edit_locked(state):

                show_parameter_lock_message(state)
                return

            if state.current_fractal == FractalType.JULIA:

                if event.key == pg.K_i:
                    state.julia_cy += 0.005

                elif event.key == pg.K_k:
                    state.julia_cy -= 0.005

                elif event.key == pg.K_j:
                    state.julia_cx -= 0.005

                elif event.key == pg.K_l:
                    state.julia_cx += 0.005

                state.needs_update = True

            elif state.current_fractal == FractalType.MANDELBROT:

                if event.key == pg.K_i:
                    state.mandelbrot_zy += 0.005

                elif event.key == pg.K_k:
                    state.mandelbrot_zy -= 0.005

                elif event.key == pg.K_j:
                    state.mandelbrot_zx -= 0.005

                elif event.key == pg.K_l:
                    state.mandelbrot_zx += 0.005

                state.needs_update = True
        
    @staticmethod
    def handle_mouse(state, event, audio_manager):

        mx, my = pg.mouse.get_pos()

        if state.benchmark_results_open:
            return

        if state.preset_name_input_active:
            return

        # ========================================================
        # MENU
        # ========================================================

        if state.menu_open:

            if event.button != 1:
                return

            panel = pg.Rect(
                80,
                120,
                CONFIG.width - 160,
                CONFIG.height - 240
            )

            category_options = get_category_options(
                state,
                state.submenu_open
            )

            if category_options:

                y = panel.y + 30

                for option in category_options:

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    if rect.collidepoint(mx, my):
                        InputHandler.activate_menu_option(
                            state,
                            option,
                            audio_manager
                        )
                        return

                    y += 52

                return

            if state.submenu_open == "Mandelbrot Presets":

                y = panel.y + 30

                add_rect = pg.Rect(
                    panel.x + 30,
                    y,
                    panel.width - 60,
                    42
                )

                if add_rect.collidepoint(mx, my):
                    begin_preset_name_input(
                        state,
                        "mandelbrot"
                    )
                    audio_manager.play_click()
                    return

                y += 52

                for i, preset in enumerate(
                    get_combined_mandelbrot_presets(state)
                ):

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    delete_rect = pg.Rect(
                        rect.right - 52,
                        rect.y + 6,
                        36,
                        30
                    )

                    if (
                        preset.get("custom")
                        and delete_rect.collidepoint(mx, my)
                    ):
                        delete_custom_preset(
                            state,
                            "mandelbrot",
                            i
                        )
                        audio_manager.play_click()
                        return

                    if rect.collidepoint(mx, my):

                        if is_parameter_edit_locked(state):
                            show_parameter_lock_message(state)
                            audio_manager.play_click()
                            return

                        apply_mandelbrot_preset(
                            state,
                            preset,
                            i
                        )

                        audio_manager.play_click()

                    y += 52

                return

            if state.submenu_open == "Julia Presets":

                y = panel.y + 30

                add_rect = pg.Rect(
                    panel.x + 30,
                    y,
                    panel.width - 60,
                    42
                )

                if add_rect.collidepoint(mx, my):
                    begin_preset_name_input(
                        state,
                        "julia"
                    )
                    audio_manager.play_click()
                    return

                y += 52

                for i, preset in enumerate(
                    get_combined_julia_presets(state)
                ):

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    delete_rect = pg.Rect(
                        rect.right - 52,
                        rect.y + 6,
                        36,
                        30
                    )

                    if (
                        preset.get("custom")
                        and delete_rect.collidepoint(mx, my)
                    ):
                        delete_custom_preset(
                            state,
                            "julia",
                            i
                        )
                        audio_manager.play_click()
                        return

                    if rect.collidepoint(mx, my):

                        if is_parameter_edit_locked(state):
                            show_parameter_lock_message(state)
                            audio_manager.play_click()
                            return

                        apply_julia_preset(
                            state,
                            preset,
                            i
                        )

                        audio_manager.play_click()

                    y += 52

                return

            if state.submenu_open == "Power":

                y = panel.y + 30

                for power in POWER_VALUES:

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    if rect.collidepoint(mx, my):

                        if is_parameter_edit_locked(state):
                            show_parameter_lock_message(state)
                            audio_manager.play_click()
                            return

                        state.fractal_power = power
                        state.needs_update = True

                        audio_manager.play_click()

                    y += 52

                return

            if state.submenu_open == "Fractal Type":

                y = panel.y + 30

                fractals = [
                    (FractalType.MANDELBROT, "Mandelbrot"),
                    (FractalType.JULIA, "Julia Set"),
                    (FractalType.BURNING_SHIP, "Burning Ship"),
                ]

                for fractal_type, label in fractals:

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        50
                    )

                    if rect.collidepoint(mx, my):

                        previous_fractal = (
                            state.current_fractal
                        )

                        state.current_fractal = fractal_type

                        if (
                            previous_fractal != FractalType.MANDELBROT
                            and fractal_type == FractalType.MANDELBROT
                        ):

                            new_state = create_initial_state()

                            state.xmin = new_state.xmin
                            state.xmax = new_state.xmax
                            state.ymin = new_state.ymin
                            state.ymax = new_state.ymax

                        state.needs_update = True

                        if previous_fractal != fractal_type:
                            unlock_achievement(
                                state,
                                "fractal_switcher"
                            )

                        audio_manager.play_click()

                    y += 62

                return

            if state.submenu_open == "Palette":

                y = panel.y + 30

                for i, palette in enumerate(PALETTES):

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    if rect.collidepoint(mx, my):

                        previous_palette = state.current_palette

                        state.current_palette = i
                        state.needs_update = True

                        if previous_palette != i:
                            unlock_achievement(
                                state,
                                "palette_swapper"
                            )

                        audio_manager.play_click()

                    y += 52

                return

            if state.submenu_open == "Game Resolution":

                y = panel.y + 30

                for width, height in GAME_RESOLUTION_PRESETS:

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    if rect.collidepoint(mx, my):
                        apply_game_resolution(
                            state,
                            width,
                            height
                        )
                        audio_manager.play_click()
                        return

                    y += 52

            if state.submenu_open == "PNG Resolution":

                y = panel.y + 30

                for width, height in PNG_RESOLUTION_PRESETS:

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    if rect.collidepoint(mx, my):
                        set_png_resolution(
                            width,
                            height
                        )
                        audio_manager.play_click()
                        return

                    y += 52

            if state.submenu_open == "GIF Resolution":

                y = panel.y + 30

                for width, height in GIF_RESOLUTION_PRESETS:

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    if rect.collidepoint(mx, my):
                        set_gif_resolution(
                            width,
                            height
                        )
                        audio_manager.play_click()
                        return

                    y += 52

            if state.submenu_open == "Audio Settings":

                sliders = [
                    (
                        "music_volume",
                        panel.y + 110
                    ),
                    (
                        "ui_volume",
                        panel.y + 220
                    ),
                ]

                for key, slider_y in sliders:

                    slider_rect = pg.Rect(
                        panel.x + 260,
                        slider_y + 6,
                        panel.width - 360,
                        22
                    )

                    hit_rect = slider_rect.inflate(
                        20,
                        34
                    )

                    if hit_rect.collidepoint(mx, my):
                        value = (
                            mx - slider_rect.x
                        ) / slider_rect.width

                        set_volume(
                            audio_manager,
                            key,
                            value
                        )

                        audio_manager.play_click()
                        return

                return

            if state.submenu_open == "Music Volume":

                change_volume(
                    audio_manager,
                    "music_volume",
                    0.05
                )
                audio_manager.play_click()
                return

            if state.submenu_open == "UI Volume":

                change_volume(
                    audio_manager,
                    "ui_volume",
                    0.05
                )
                audio_manager.play_click()
                return

            if state.submenu_open == "Show Performance Info":

                toggle_performance_info()
                audio_manager.play_click()
                return

            start_y = 240

            for i, option in enumerate(get_menu_options(state)):

                rect = pg.Rect(
                    CONFIG.width // 2 - 250,
                    start_y + i * 70,
                    500,
                    55
                )

                if rect.collidepoint(mx, my):

                    state.menu_index = i

                    InputHandler.activate_menu_option(
                        state,
                        option,
                        audio_manager
                    )

            return

        # ========================================================
        # NORMAL
        # ========================================================

        if event.button == 1:

            Camera.center_on_pixel(
                state,
                mx,
                my
            )

        elif event.button == 4:

            Camera.zoom(state, 1)
            unlock_achievement(state, "first_zoom")

        elif event.button == 5:

            Camera.zoom(state, -1)
            unlock_achievement(state, "first_zoom")

# ============================================================================
# UI
# ============================================================================


class UI:

    def __init__(self):

        self.font = pg.font.SysFont(
            CONFIG.font_name,
            CONFIG.font_size
        )

    def draw(self, screen, state):

        if not state.show_ui:

            if state.tutorial_active:
                self.draw_tutorial_overlay(screen, state)

            return

        palette_name = PALETTES[
            state.current_palette
        ]["name"]

        lines = [
            f"Fractal: {state.current_fractal.value}",
        ]
        
        if state.current_fractal == FractalType.MANDELBROT:

            lines += [
                f"Mandelbrot Z Real: {state.mandelbrot_zx:.6f}",
                f"Mandelbrot Z Imag: {state.mandelbrot_zy:.6f}",
                f"Mandelbrot Preset: {state.mandelbrot_preset_name}",
            ]

        if state.current_fractal == FractalType.JULIA:

            lines += [
                f"Julia C Real: {state.julia_cx:.6f}",
                f"Julia C Imag: {state.julia_cy:.6f}",
                f"Julia Preset: {state.julia_preset_name}",
            ]

        if state.current_fractal in (
            FractalType.MANDELBROT,
            FractalType.JULIA
        ):

            lines += [
                f"Power: {state.fractal_power}",
            ]
        
        if is_parameter_edit_locked(state):

            lines += [
                "Parameter editing locked: zoom out to change fractal parameters",
            ]

        lines += [
            f"Iterations: {state.max_iter}",
            f"Palette: {palette_name}",
        ]

        if CONFIG.show_performance_info:

            lines += [
                f"Render: {state.last_render_time_ms:.2f} ms",
                f"FPS: {state.last_render_fps:.1f}",
            ]

        lines += [
            "",
            "ESC = menu"
        ]
        

        y = 10

        for line in lines:

            text = self.font.render(
                line,
                True,
                (255, 255, 255)
            )

            screen.blit(text, (10, y))

            y += 30

        now = time.time()

        if (
            state.export_message and
            now - state.export_message_timer < 5.0
        ):

            text = self.font.render(
                state.export_message,
                True,
                (255, 255, 255)
            )

            screen.blit(
                text,
                (20, CONFIG.height - 40)
            )

        if state.tutorial_active:
            self.draw_tutorial_overlay(screen, state)

    def draw_tutorial_overlay(self, screen, state):

        overlay = pg.Surface(
            (CONFIG.width, CONFIG.height),
            pg.SRCALPHA
        )

        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        panel = pg.Rect(
            CONFIG.width // 2 - 550,
            CONFIG.height - 310,
            1100,
            230
        )

        pg.draw.rect(
            screen,
            (20, 20, 24),
            panel,
            border_radius=16
        )

        pg.draw.rect(
            screen,
            (90, 90, 140),
            panel,
            width=2,
            border_radius=16
        )

        step_index = max(
            0,
            min(state.tutorial_step_index, len(TUTORIAL_STEPS) - 1)
        )

        step = TUTORIAL_STEPS[step_index]

        title = step.get(
            "title",
            "Tutorial"
        )

        body, required = get_tutorial_step_text(state)

        title_text = self.font.render(
            f"Tutorial {step_index + 1}/{len(TUTORIAL_STEPS)}: {title}",
            True,
            (255, 255, 255)
        )

        screen.blit(
            title_text,
            (panel.x + 30, panel.y + 25)
        )

        body_text = self.font.render(
            body,
            True,
            (220, 220, 220)
        )

        screen.blit(
            body_text,
            (panel.x + 30, panel.y + 78)
        )

        required_text = self.font.render(
            required,
            True,
            (255, 255, 255)
        )

        screen.blit(
            required_text,
            (panel.x + 30, panel.y + 118)
        )

        hint_text = self.font.render(
            "ESC skips tutorial",
            True,
            (190, 190, 210)
        )

        screen.blit(
            hint_text,
            (panel.x + 30, panel.y + 165)
        )

    def draw_benchmark_results_overlay(self, screen, state):

        overlay = pg.Surface(
            (CONFIG.width, CONFIG.height),
            pg.SRCALPHA
        )

        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        panel_width = min(
            900,
            CONFIG.width - 120
        )

        panel_height = min(
            560,
            CONFIG.height - 120
        )

        panel = pg.Rect(
            CONFIG.width // 2 - panel_width // 2,
            CONFIG.height // 2 - panel_height // 2,
            panel_width,
            panel_height
        )

        pg.draw.rect(
            screen,
            (20, 20, 24),
            panel,
            border_radius=16
        )

        pg.draw.rect(
            screen,
            (90, 90, 140),
            panel,
            width=2,
            border_radius=16
        )

        y = panel.y + 40

        for index, line in enumerate(state.benchmark_result_lines):

            font = self.font if index == 0 else self.font

            text = font.render(
                line,
                True,
                (255, 255, 255)
            )

            screen.blit(
                text,
                (panel.x + 40, y)
            )

            y += 42


# ============================================================================
# MENU UI
# ============================================================================


class MenuUI:

    def __init__(self):

        self.font = pg.font.SysFont(
            CONFIG.font_name,
            34
        )

        self.small_font = pg.font.SysFont(
            CONFIG.font_name,
            22
        )

    def update_hover_sound(
        self,
        state,
        audio_manager,
        hovered_item
    ):

        if hovered_item != state.hovered_menu_item:

            state.hovered_menu_item = hovered_item

            if hovered_item:
                audio_manager.play_hover()

    def draw(self, screen, state, audio_manager):

        overlay = pg.Surface(
            (CONFIG.width, CONFIG.height),
            pg.SRCALPHA
        )

        overlay.fill((0, 0, 0, 180))

        screen.blit(overlay, (0, 0))

        title = self.font.render(
            "SETTINGS MENU",
            True,
            (255, 255, 255)
        )

        screen.blit(title, (50, 50))

        start_y = 240

        mouse_x, mouse_y = pg.mouse.get_pos()

        hovered_item = None

        for i, option in enumerate(get_menu_options(state)):

            rect = pg.Rect(
                CONFIG.width // 2 - 250,
                start_y + i * 70,
                500,
                55
            )

            hovered = rect.collidepoint(
                mouse_x,
                mouse_y
            )

            if hovered:
                hovered_item = f"menu:{option}"

            selected = (
                hovered or
                i == state.menu_index
            )

            bg = (
                (80, 80, 120)
                if selected
                else
                (40, 40, 40)
            )

            pg.draw.rect(
                screen,
                bg,
                rect,
                border_radius=10
            )

            text = self.font.render(
                option,
                True,
                (255, 255, 255)
            )

            screen.blit(
                text,
                (
                    rect.x + 20,
                    rect.y + 10
                )
            )

        submenu_hovered_item = self.draw_submenu(screen, state)

        if submenu_hovered_item:
            hovered_item = submenu_hovered_item

        if state.preset_name_input_active:
            self.draw_preset_name_modal(screen, state)

        self.update_hover_sound(
            state,
            audio_manager,
            hovered_item
        )

    def draw_preset_name_modal(self, screen, state):

        modal = pg.Rect(
            CONFIG.width // 2 - 330,
            CONFIG.height // 2 - 120,
            660,
            220
        )

        pg.draw.rect(
            screen,
            (20, 20, 20),
            modal,
            border_radius=16
        )

        pg.draw.rect(
            screen,
            (90, 90, 140),
            modal,
            width=2,
            border_radius=16
        )

        title = self.font.render(
            "Add Current Preset",
            True,
            (255, 255, 255)
        )

        screen.blit(
            title,
            (modal.x + 30, modal.y + 25)
        )

        input_rect = pg.Rect(
            modal.x + 30,
            modal.y + 85,
            modal.width - 60,
            46
        )

        pg.draw.rect(
            screen,
            (35, 35, 35),
            input_rect,
            border_radius=8
        )

        text = self.small_font.render(
            state.preset_name_input + "_",
            True,
            (255, 255, 255)
        )

        screen.blit(
            text,
            (input_rect.x + 12, input_rect.y + 12)
        )

        hint = self.small_font.render(
            "Enter = save    Escape = cancel",
            True,
            (220, 220, 220)
        )

        screen.blit(
            hint,
            (modal.x + 30, modal.y + 155)
        )

    def draw_option_rows(
        self,
        screen,
        rows,
        selected_index,
        hover_prefix,
        panel,
        y
    ):

        mouse_x, mouse_y = pg.mouse.get_pos()
        hovered_item = None

        for i, label in enumerate(rows):

            rect = pg.Rect(
                panel.x + 30,
                y,
                panel.width - 60,
                42
            )

            hovered = rect.collidepoint(
                mouse_x,
                mouse_y
            )

            if hovered:
                hovered_item = f"{hover_prefix}:{i}"

            selected = (
                i == selected_index
            )

            bg = (
                (80, 80, 120)
                if selected or hovered
                else
                (35, 35, 35)
            )

            pg.draw.rect(
                screen,
                bg,
                rect,
                border_radius=8
            )

            text = self.small_font.render(
                label,
                True,
                (255, 255, 255)
            )

            screen.blit(
                text,
                (
                    rect.x + 10,
                    rect.y + 10
                )
            )

            y += 52

        return hovered_item

    def draw_volume_slider(
        self,
        screen,
        panel,
        y,
        label,
        value,
        hover_id
    ):

        mouse_x, mouse_y = pg.mouse.get_pos()

        slider_rect = pg.Rect(
            panel.x + 260,
            y + 6,
            panel.width - 360,
            22
        )

        hit_rect = slider_rect.inflate(
            20,
            34
        )

        hovered = hit_rect.collidepoint(
            mouse_x,
            mouse_y
        )

        label_text = self.small_font.render(
            f"{label}: {int(value * 100):3d}%",
            True,
            (255, 255, 255)
        )

        screen.blit(
            label_text,
            (panel.x + 30, y)
        )

        bg = (
            (80, 80, 120)
            if hovered
            else (45, 45, 45)
        )

        pg.draw.rect(
            screen,
            bg,
            slider_rect,
            border_radius=10
        )

        fill_rect = pg.Rect(
            slider_rect.x,
            slider_rect.y,
            int(slider_rect.width * clamp_volume(value)),
            slider_rect.height
        )

        pg.draw.rect(
            screen,
            (120, 120, 180),
            fill_rect,
            border_radius=10
        )

        knob_x = (
            slider_rect.x +
            int(slider_rect.width * clamp_volume(value))
        )

        pg.draw.circle(
            screen,
            (230, 230, 255),
            (knob_x, slider_rect.centery),
            15
        )

        if hovered:
            return hover_id

        return None

    def draw_submenu(self, screen, state):

        if not state.submenu_open:
            return None

        panel = pg.Rect(
            80,
            120,
            CONFIG.width - 160,
            CONFIG.height - 240
        )

        pg.draw.rect(
            screen,
            (18, 18, 18),
            panel,
            border_radius=16
        )

        y = panel.y + 30

        lines = []

        mouse_x, mouse_y = pg.mouse.get_pos()
        hovered_item = None

        title = self.font.render(
            state.submenu_open,
            True,
            (255, 255, 255)
        )

        screen.blit(
            title,
            (panel.x + 30, panel.y - 60)
        )

        category_options = get_category_options(
            state,
            state.submenu_open
        )

        if category_options:

            rows = []

            for option in category_options:

                label = option

                if option == "Show Performance Info":
                    label = (
                        "Show Performance Info: " +
                        ("ON" if CONFIG.show_performance_info else "OFF")
                    )

                rows.append(label)

            return self.draw_option_rows(
                screen,
                rows,
                -1,
                f"submenu:{state.submenu_open}",
                panel,
                y
            )

        if state.submenu_open == "Audio Settings":

            music_hover = self.draw_volume_slider(
                screen,
                panel,
                y + 80,
                "Music Volume",
                CONFIG.music_volume,
                "submenu:audio:music"
            )

            ui_hover = self.draw_volume_slider(
                screen,
                panel,
                y + 190,
                "UI Volume",
                CONFIG.ui_volume,
                "submenu:audio:ui"
            )

            return music_hover or ui_hover

        if state.submenu_open == "Palette":

            rows = [
                palette["name"]
                for palette in PALETTES
            ]

            return self.draw_option_rows(
                screen,
                rows,
                state.current_palette,
                "submenu:palette",
                panel,
                y
            )

        elif state.submenu_open == "Power":

            if is_parameter_edit_locked(state):

                warning = self.small_font.render(
                    "Parameter editing locked: zoom out to change power",
                    True,
                    (255, 220, 120)
                )

                screen.blit(
                    warning,
                    (panel.x + 30, y)
                )

                y += 42

            rows = [
                f"Power {power}"
                for power in POWER_VALUES
            ]

            selected = POWER_VALUES.index(
                state.fractal_power
            )

            return self.draw_option_rows(
                screen,
                rows,
                selected,
                "submenu:power",
                panel,
                y
            )

        elif state.submenu_open == "Iterations":

            lines = [
                f"Current Iterations: {state.max_iter}",
                "",
                "Q = decrease",
                "E = increase"
            ]

        elif state.submenu_open == "Game Resolution":

            rows = [
                f"{width}x{height}"
                for width, height in GAME_RESOLUTION_PRESETS
            ]

            selected = GAME_RESOLUTION_PRESETS.index(
                (CONFIG.base_width, CONFIG.base_height)
            )

            return self.draw_option_rows(
                screen,
                rows,
                selected,
                "submenu:game_resolution",
                panel,
                y
            )

        elif state.submenu_open == "PNG Resolution":

            rows = [
                f"{width}x{height}"
                for width, height in PNG_RESOLUTION_PRESETS
            ]

            selected = PNG_RESOLUTION_PRESETS.index(
                (
                    CONFIG.png_export_width,
                    CONFIG.png_export_height
                )
            )

            return self.draw_option_rows(
                screen,
                rows,
                selected,
                "submenu:png_resolution",
                panel,
                y
            )

        elif state.submenu_open == "GIF Resolution":

            rows = [
                f"{width}x{height}"
                for width, height in GIF_RESOLUTION_PRESETS
            ]

            selected = GIF_RESOLUTION_PRESETS.index(
                (
                    CONFIG.gif_export_width,
                    CONFIG.gif_export_height
                )
            )

            return self.draw_option_rows(
                screen,
                rows,
                selected,
                "submenu:gif_resolution",
                panel,
                y
            )

        elif state.submenu_open == "Show Performance Info":

            lines = [
                "Show Performance Info: " +
                ("ON" if CONFIG.show_performance_info else "OFF"),
                "",
                "Enter/LEFT/RIGHT/click = toggle"
            ]

        elif state.submenu_open == "Fractal Type":

            fractals = [
                (FractalType.MANDELBROT, "Mandelbrot"),
                (FractalType.JULIA, "Julia Set"),
                (FractalType.BURNING_SHIP, "Burning Ship"),
            ]

            rows = [
                label
                for fractal_type, label in fractals
            ]

            selected = [
                fractal_type
                for fractal_type, label in fractals
            ].index(state.current_fractal)

            return self.draw_option_rows(
                screen,
                rows,
                selected,
                "submenu:fractal",
                panel,
                y
            )

        elif state.submenu_open == "Mandelbrot Presets":

            if is_parameter_edit_locked(state):

                warning = self.small_font.render(
                    "Parameter editing locked: zoom out to change presets",
                    True,
                    (255, 220, 120)
                )

                screen.blit(
                    warning,
                    (panel.x + 30, y)
                )

                y += 42

            add_rect = pg.Rect(
                panel.x + 30,
                y,
                panel.width - 60,
                42
            )

            add_hovered = add_rect.collidepoint(
                mouse_x,
                mouse_y
            )

            if add_hovered:
                hovered_item = "submenu:mandelbrot:add"

            pg.draw.rect(
                screen,
                (70, 70, 110) if add_hovered else (35, 35, 35),
                add_rect,
                border_radius=8
            )

            text = self.small_font.render(
                "Add Current Preset",
                True,
                (255, 255, 255)
            )

            screen.blit(
                text,
                (add_rect.x + 10, add_rect.y + 10)
            )

            y += 52

            for i, preset in enumerate(
                get_combined_mandelbrot_presets(state)
            ):

                name = preset["name"]
                zx = preset["mandelbrot_zx"]
                zy = preset["mandelbrot_zy"]

                rect = pg.Rect(
                    panel.x + 30,
                    y,
                    panel.width - 60,
                    42
                )

                delete_rect = pg.Rect(
                    rect.right - 52,
                    rect.y + 6,
                    36,
                    30
                )

                hovered = rect.collidepoint(
                    mouse_x,
                    mouse_y
                )

                if hovered:
                    hovered_item = f"submenu:mandelbrot:{i}"

                selected = (
                    i == state.mandelbrot_preset_index
                )

                bg = (
                    (80, 80, 120)
                    if selected
                    else
                    (35, 35, 35)
                )

                pg.draw.rect(
                    screen,
                    bg,
                    rect,
                    border_radius=8
                )

                label = f"{name}  ({zx:.6f}, {zy:.6f})"

                if preset.get("custom"):
                    label += "  [custom]"

                text = self.small_font.render(
                    label,
                    True,
                    (255, 255, 255)
                )

                screen.blit(
                    text,
                    (
                        rect.x + 10,
                        rect.y + 10
                    )
                )

                if preset.get("custom"):
                    delete_hovered = delete_rect.collidepoint(
                        mouse_x,
                        mouse_y
                    )

                    if delete_hovered:
                        hovered_item = f"submenu:mandelbrot:delete:{i}"

                    pg.draw.rect(
                        screen,
                        (110, 50, 50) if delete_hovered else (70, 40, 40),
                        delete_rect,
                        border_radius=6
                    )

                    delete_text = self.small_font.render(
                        "X",
                        True,
                        (255, 255, 255)
                    )

                    screen.blit(
                        delete_text,
                        (delete_rect.x + 11, delete_rect.y + 4)
                    )

                y += 52

            return hovered_item

        elif state.submenu_open == "Julia Presets":

            if is_parameter_edit_locked(state):

                warning = self.small_font.render(
                    "Parameter editing locked: zoom out to change presets",
                    True,
                    (255, 220, 120)
                )

                screen.blit(
                    warning,
                    (panel.x + 30, y)
                )

                y += 42

            add_rect = pg.Rect(
                panel.x + 30,
                y,
                panel.width - 60,
                42
            )

            add_hovered = add_rect.collidepoint(
                mouse_x,
                mouse_y
            )

            if add_hovered:
                hovered_item = "submenu:julia:add"

            pg.draw.rect(
                screen,
                (70, 70, 110) if add_hovered else (35, 35, 35),
                add_rect,
                border_radius=8
            )

            text = self.small_font.render(
                "Add Current Preset",
                True,
                (255, 255, 255)
            )

            screen.blit(
                text,
                (add_rect.x + 10, add_rect.y + 10)
            )

            y += 52

            for i, preset in enumerate(
                get_combined_julia_presets(state)
            ):

                name = preset["name"]
                cx = preset["julia_cx"]
                cy = preset["julia_cy"]

                rect = pg.Rect(
                    panel.x + 30,
                    y,
                    panel.width - 60,
                    42
                )

                delete_rect = pg.Rect(
                    rect.right - 52,
                    rect.y + 6,
                    36,
                    30
                )

                hovered = rect.collidepoint(
                    mouse_x,
                    mouse_y
                )

                if hovered:
                    hovered_item = f"submenu:julia:{i}"

                selected = (
                    i == state.julia_preset_index
                )

                bg = (
                    (80, 80, 120)
                    if selected
                    else
                    (35, 35, 35)
                )

                pg.draw.rect(
                    screen,
                    bg,
                    rect,
                    border_radius=8
                )

                label = f"{name}  ({cx:.6f}, {cy:.6f})"

                if preset.get("custom"):
                    label += "  [custom]"

                text = self.small_font.render(
                    label,
                    True,
                    (255, 255, 255)
                )

                screen.blit(
                    text,
                    (
                        rect.x + 10,
                        rect.y + 10
                    )
                )

                if preset.get("custom"):
                    delete_hovered = delete_rect.collidepoint(
                        mouse_x,
                        mouse_y
                    )

                    if delete_hovered:
                        hovered_item = f"submenu:julia:delete:{i}"

                    pg.draw.rect(
                        screen,
                        (110, 50, 50) if delete_hovered else (70, 40, 40),
                        delete_rect,
                        border_radius=6
                    )

                    delete_text = self.small_font.render(
                        "X",
                        True,
                        (255, 255, 255)
                    )

                    screen.blit(
                        delete_text,
                        (delete_rect.x + 11, delete_rect.y + 4)
                    )

                y += 52

            return hovered_item

        elif state.submenu_open == "Achievements":

            for achievement_id, data in ACHIEVEMENTS.items():

                name, description = data

                unlocked = state.user_progress.get(
                    "achievements",
                    {}
                ).get(achievement_id, False)

                rect = pg.Rect(
                    panel.x + 30,
                    y,
                    panel.width - 60,
                    58
                )

                pg.draw.rect(
                    screen,
                    (55, 70, 55) if unlocked else (35, 35, 35),
                    rect,
                    border_radius=8
                )

                title = (
                    f"{name} - Unlocked"
                    if unlocked
                    else f"{name} - Locked"
                )

                title_text = self.small_font.render(
                    title,
                    True,
                    (255, 255, 255)
                )

                desc_text = self.small_font.render(
                    description if unlocked else "Locked",
                    True,
                    (210, 210, 210)
                )

                screen.blit(
                    title_text,
                    (rect.x + 10, rect.y + 8)
                )

                screen.blit(
                    desc_text,
                    (rect.x + 10, rect.y + 32)
                )

                y += 68

            return hovered_item

        elif state.submenu_open == "Controls":

            lines = [
                "WASD = move",
                "Mouse wheel = zoom",
                "Left click = center",
                "Q/E = iterations",
                "R/F = palettes",
                "I/J/K/L = edit Julia or Mandelbrot parameters",
                "P = export PNG",
                "Export Settings -> Export GIF = zoom animation"
            ]

        for line in lines:

            text = self.small_font.render(
                line,
                True,
                (220, 220, 220)
            )

            screen.blit(
                text,
                (panel.x + 40, y)
            )

            y += 34

        return hovered_item

# ============================================================================
# RENDERER
# ============================================================================


class Renderer:

    def __init__(self, screen):

        self.screen = screen

    def render_frame(
        self,
        state,
        ui,
        menu_ui,
        audio_manager
    ):

        self.screen = pg.display.get_surface()

        if state.needs_update:

            t0 = time.perf_counter()

            state.surface = compute_surface(
                state,
                CONFIG.width,
                CONFIG.height
            )

            dt = time.perf_counter() - t0

            # ====================================================
            # STORE RENDER STATS
            # ====================================================

            state.last_render_time_ms = dt * 1000.0

            if dt > 0.0:
                state.last_render_fps = 1.0 / dt
            else:
                state.last_render_fps = 0.0

            state.needs_update = False

            if CONFIG.show_performance_info:
                pg.display.set_caption(
                    f"{CONFIG.window_title} "
                    f"| render {state.last_render_time_ms:.1f} ms "
                    f"| {state.last_render_fps:.1f} FPS"
                )
            else:
                pg.display.set_caption(
                    CONFIG.window_title
                )

        if state.surface:

            self.screen.blit(
                state.surface,
                (0, 0)
            )

        ui.draw(self.screen, state)

        if state.menu_open:

            menu_ui.draw(
                self.screen,
                state,
                audio_manager
            )

        if state.benchmark_results_open:

            ui.draw_benchmark_results_overlay(
                self.screen,
                state
            )

# ============================================================================
# APPLICATION
# ============================================================================


class Application:

    def __init__(self):

        apply_user_settings_to_config(
            load_user_settings()
        )

        pg.init()

        self.screen = pg.display.set_mode(
            (
                CONFIG.width,
                CONFIG.height
            )
        )

        pg.display.set_caption(
            CONFIG.window_title
        )

        self.clock = pg.time.Clock()

        self.audio_manager = AudioManager()
        self.audio_manager.start_music()

        self.state = create_initial_state()
        self.state.custom_presets = load_user_presets()
        self.state.user_progress = load_user_progress()

        if not self.state.user_progress.get("tutorial_completed", False):
            self.state.tutorial_active = True

        self.ui = UI()

        self.menu_ui = MenuUI()

        self.renderer = Renderer(
            self.screen
        )

    def run(self):

        while self.state.running:

            InputHandler.process_events(
                self.state,
                self.audio_manager
            )

            self.renderer.render_frame(
                self.state,
                self.ui,
                self.menu_ui,
                self.audio_manager
            )

            pg.display.flip()

            self.clock.tick(
                CONFIG.target_fps
            )

        pg.quit()

# ============================================================================
# ENTRY
# ============================================================================


def main():

    app = Application()

    app.run()


if __name__ == "__main__":
    main()