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

import sys
import time

from datetime import datetime
from dataclasses import dataclass
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

    #easy way to change the resolution, keeping the aspect ratio 16:9
    upscale: int = 1

    base_width: int = 1920
    base_height: int = 1080

    export_width: int = 3840
    export_height: int = 2160

    target_fps: int = 60

    base_move_speed: float = 0.05

    font_name: str = "consolas"
    font_size: int = 24

    window_title: str = "GPU Mandelbrot Explorer"

    @property
    def width(self):
        return int(self.base_width * self.upscale)

    @property
    def height(self):
        return int(self.base_height * self.upscale)


CONFIG = AppConfig()

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
# MENUS
# ============================================================================

BASE_MENU_OPTIONS = [
    "Continue",
    "Palettes",
    "Iterations",
    "Export Resolution",
    "Fractal Type",
    "Controls",
    "Fractal Information",
    "Exit"
]

def get_menu_options(state):
    """ Builds the active menu option list.
    Some menu entries are conditionally inserted depending
    on the currently selected fractal type.
    
    Args:
        state: Current application state.
    
    Returns:
        Ordered list of menu entries.
    """


    options = BASE_MENU_OPTIONS.copy()

    if state.current_fractal == FractalType.JULIA:

        options.insert(
            options.index("Controls"),
            "Julia Presets"
        )

    return options

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

    running: bool = True

    # Indicates whether the fractal surface must be re-rendered.
    needs_update: bool = True

    show_ui: bool = True

    menu_open: bool = False
    submenu_open: str | None = None
    menu_index: int = 0

    surface: pg.Surface | None = None

    # --------------------------------------------------------
    # Export State
    # --------------------------------------------------------
    exporting: bool = False
    export_message: str = ""
    export_message_timer: float = 0.0

    # --------------------------------------------------------
    # Render Performance Statistics
    # --------------------------------------------------------
    last_render_time_ms: float = 0.0
    last_render_fps: float = 0.0

    # --------------------------------------------------------
    # Julia Fractal Configuration
    # --------------------------------------------------------
    julia_cx: float = -0.8
    julia_cy: float = 0.156

    julia_preset_index: int = 0

    fractal_info_scroll: int = 0


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

    if (fractal_type == 1)
    {
        // JULIA

        z0r = cr;
        z0i = ci;

        c_r = dd_set(julia_cx);
        c_i = dd_set(julia_cy);
    }
    else
    {
        // MANDELBROT / BURNING SHIP

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

        dd zr_new = dd_add(
            dd_sub(zr2, zi2),
            c_r
        );

        dd zi_new = dd_add(
            two_zrzi,
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


def render_fractal_gpu(
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

            gpu_buffer,
        )
    )

    rgb_hw3 = colorize_gpu(
        gpu_buffer,
        state.max_iter,
        state.current_palette,
        width,
        height
    )

    return rgb_hw3


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


def export_fractal_png(state):

    try:

        state.exporting = True

        state.export_message = (
            "Rendering export..."
        )

        rgb_hw3 = render_fractal_gpu(
            state,
            CONFIG.export_width,
            CONFIG.export_height
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

        filename = (
            f"fractal_{timestamp}.png"
        )

        pg.image.save(surface, filename)

        state.exporting = False

        state.export_message = (
            f"Exported: {filename}"
        )

        state.export_message_timer = time.time()

    except Exception as e:

        state.exporting = False

        state.export_message = (
            f"EXPORT ERROR: {e}"
        )

        state.export_message_timer = time.time()

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

# ============================================================================
# INPUT
# ============================================================================


class InputHandler:

    @staticmethod
    def process_events(state):

        for event in pg.event.get():

            if event.type == pg.QUIT:
                state.running = False

            elif event.type == pg.KEYDOWN:
                InputHandler.handle_keydown(
                    state,
                    event
                )

            elif event.type == pg.MOUSEBUTTONDOWN:
                InputHandler.handle_mouse(
                    state,
                    event
                )

    @staticmethod
    def activate_menu_option(state, option):

        if option == "Continue":

            state.menu_open = False
            state.submenu_open = None

        elif option == "Exit":

            state.running = False

        else:

            state.submenu_open = option

    @staticmethod
    def handle_keydown(state, event):

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

        if state.submenu_open == "Export Resolution":

            if event.key == pg.K_LEFT:
                CONFIG.export_width = max(
                    256,
                    CONFIG.export_width - 256
                )

            elif event.key == pg.K_RIGHT:
                CONFIG.export_width += 256

            elif event.key == pg.K_DOWN:
                CONFIG.export_height = max(
                    256,
                    CONFIG.export_height - 256
                )

            elif event.key == pg.K_UP:
                CONFIG.export_height += 256

        

        if state.menu_open:

            if state.submenu_open == "Fractal Information":

                if event.key == pg.K_UP:
                    state.fractal_info_scroll += 40

                elif event.key == pg.K_DOWN:
                    state.fractal_info_scroll -= 40

                state.fractal_info_scroll = min(
                    0,
                    state.fractal_info_scroll
                )
            elif event.key == pg.K_DOWN:

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
                    option
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

        elif event.key == pg.K_f:

            state.current_palette -= 1
            state.current_palette %= len(
                PALETTES
            )

            state.needs_update = True

        elif event.key == pg.K_p:

            export_fractal_png(state)

        elif (
            state.current_fractal == FractalType.JULIA
            and event.key == pg.K_i
        ):
        
            state.julia_cy += 0.005
            state.needs_update = True
        
        elif (
            state.current_fractal == FractalType.JULIA
            and event.key == pg.K_k
        ):
        
            state.julia_cy -= 0.005
            state.needs_update = True
        
        elif (
            state.current_fractal == FractalType.JULIA
            and event.key == pg.K_j
        ):
        
            state.julia_cx -= 0.005
            state.needs_update = True
        
        elif (
            state.current_fractal == FractalType.JULIA
            and event.key == pg.K_l
        ):
        
            state.julia_cx += 0.005
            state.needs_update = True
    
        elif event.key == pg.K_i:
        
            state.julia_cy += 0.005
            state.needs_update = True
        
        elif event.key == pg.K_k:
        
            state.julia_cy -= 0.005
            state.needs_update = True
        
        elif event.key == pg.K_j:
        
            state.julia_cx -= 0.005
            state.needs_update = True
        
        elif event.key == pg.K_l:
        
            state.julia_cx += 0.005
            state.needs_update = True
        
    @staticmethod
    def handle_mouse(state, event):

        mx, my = pg.mouse.get_pos()

        # ========================================================
        # MENU
        # ========================================================

        if state.menu_open:

            if state.submenu_open == "Julia Presets":

                panel = pg.Rect(
                    80,
                    120,
                    CONFIG.width - 160,
                    CONFIG.height - 240
                )
            
                y = panel.y + 30
            
                for i, preset in enumerate(JULIA_PRESETS):
            
                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )
            
                    if rect.collidepoint(mx, my):
            
                        name, cx, cy = preset
            
                        state.julia_preset_index = i
            
                        state.julia_cx = cx
                        state.julia_cy = cy
            
                        state.current_fractal = FractalType.JULIA
            
                        state.needs_update = True
            
                    y += 52

                return

            if state.submenu_open == "Fractal Information":

                if event.button == 4:
                    state.fractal_info_scroll += 40

                elif event.button == 5:
                    state.fractal_info_scroll -= 40

                state.fractal_info_scroll = min(
                    0,
                    state.fractal_info_scroll
                )

            if state.submenu_open == "Palettes":

                panel = pg.Rect(
                    80,
                    120,
                    CONFIG.width - 160,
                    CONFIG.height - 240
                )

                y = panel.y + 30

                for i, palette in enumerate(PALETTES):

                    rect = pg.Rect(
                        panel.x + 30,
                        y,
                        panel.width - 60,
                        42
                    )

                    if rect.collidepoint(mx, my):

                        state.current_palette = i

                        state.needs_update = True

                    y += 52

            if event.button == 1:

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
                            option
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

        elif event.button == 5:

            Camera.zoom(state, -1)

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
            return

        palette_name = PALETTES[
            state.current_palette
        ]["name"]

        lines = [
            f"Fractal: {state.current_fractal.value}",
        ]
        
        if state.current_fractal == FractalType.JULIA:
        
            lines += [
                f"Julia C Real: {state.julia_cx:.6f}",
                f"Julia C Imag: {state.julia_cy:.6f}",
                f"Julia Preset: {JULIA_PRESETS[state.julia_preset_index][0]}",
            ]
        
        lines += [
            f"Iterations: {state.max_iter}",
            f"Palette: {palette_name}",
            f"Render: {state.last_render_time_ms:.2f} ms",
            f"FPS: {state.last_render_fps:.1f}",
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


# ============================================================================
# FRACTAL INFORMATION TEXT
# ============================================================================

FRACTAL_INFORMATION_TEXT = """
FRACTALS

Fractals are mathematical structures that contain repeating patterns
visible at many different scales. When zooming into a fractal,
similar shapes continue appearing infinitely.

Fractals are created using mathematics and iterative equations.
Even very simple formulas can generate extremely complex images.

MANDELBROT SET

The Mandelbrot Set is one of the most famous fractals in mathematics.

It is generated using the equation:

z = z^2 + c

Each pixel on the screen represents a complex number.
The program repeatedly applies the equation and checks whether
the values remain stable or escape to infinity.

The edge of the Mandelbrot Set contains infinite detail.
No matter how far you zoom in, new structures continue to appear.

JULIA SET

Julia Sets are closely related to the Mandelbrot Set.

Instead of changing the value of c for every pixel,
Julia Sets keep c constant and change the starting position.

Different constants create completely different Julia fractals.
Some appear connected while others split into disconnected islands.

HISTORY OF FRACTALS

Fractal-like mathematics existed long before computers,
but fractals became widely known in the 20th century.

The mathematician Benoit Mandelbrot popularized the term "fractal"
in the 1970s while studying self-similar structures in nature.

Modern computers allowed mathematicians to visualize these equations
for the first time, revealing enormous hidden complexity.

WHY FRACTALS ARE INTERESTING

Fractals combine simple mathematics with infinite complexity.

They appear in many areas of science and nature:
- coastlines
- clouds
- lightning
- plants
- galaxies
- river systems

Fractals are also important in:
- chaos theory
- computer graphics
- procedural generation
- physics
- signal processing

The Mandelbrot Set is often considered one of the most beautiful
objects in mathematics because of its endless detail and structure.

CONTROLS

Mouse Wheel  - Zoom
Left Click   - Center camera
WASD         - Move
Q / E        - Iterations
R / F        - Change palette
ESC          - Open menu
"""

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

    def draw(self, screen, state):

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

        self.draw_submenu(screen, state)

    def draw_submenu(self, screen, state):

        if not state.submenu_open:
            return

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

        if state.submenu_open == "Palettes":

            for i, palette in enumerate(PALETTES):

                rect = pg.Rect(
                    panel.x + 30,
                    y,
                    panel.width - 60,
                    42
                )

                selected = (
                    i == state.current_palette
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

                text = self.small_font.render(
                    palette["name"],
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

            return

        elif state.submenu_open == "Iterations":

            lines = [
                f"Current Iterations: {state.max_iter}",
                "",
                "Q = decrease",
                "E = increase"
            ]

        elif state.submenu_open == "Export Resolution":

            lines = [
                f"Width: {CONFIG.export_width}",
                f"Height: {CONFIG.export_height}",
                "",
                "LEFT/RIGHT = width",
                "UP/DOWN = height"
            ]
        elif state.submenu_open == "Fractal Type":
        
            mouse_x, mouse_y = pg.mouse.get_pos()
        
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
        
                hovered = rect.collidepoint(
                    mouse_x,
                    mouse_y
                )
        
                selected = (
                    fractal_type ==
                    state.current_fractal
                )
        
                bg = (
                    (90, 90, 140)
                    if (hovered or selected)
                    else
                    (35, 35, 35)
                )
        
                pg.draw.rect(
                    screen,
                    bg,
                    rect,
                    border_radius=10
                )
        
                text = self.small_font.render(
                    label,
                    True,
                    (255, 255, 255)
                )
        
                screen.blit(
                    text,
                    (
                        rect.x + 15,
                        rect.y + 12
                    )
                )
        
                if (
                    pg.mouse.get_pressed()[0]
                    and hovered
                ):
        
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
        
                y += 62
        
            return


        elif state.submenu_open == "Julia Presets":

            for i, preset in enumerate(JULIA_PRESETS):

                name, cx, cy = preset

                rect = pg.Rect(
                    panel.x + 30,
                    y,
                    panel.width - 60,
                    42
                )

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

                text = self.small_font.render(
                    f"{name}  ({cx:.6f}, {cy:.6f})",
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

            return


        elif state.submenu_open == "Controls":

            lines = [
                "WASD = move",
                "Mouse wheel = zoom",
                "Left click = center",
                "Q/E = iterations",
                "R/F = palettes",
                "IJKL = Julia constant",
                "P = export PNG"
            ]

        elif state.submenu_open == "Fractal Information":

            title = self.font.render(
                "FRACTAL INFORMATION",
                True,
                (255, 255, 255)
            )

            screen.blit(
                title,
                (panel.x + 30, panel.y + 20)
            )

            back_rect = pg.Rect(
                panel.right - 140,
                panel.y + 20,
                100,
                40
            )

            pg.draw.rect(
                screen,
                (70, 70, 110),
                back_rect,
                border_radius=8
            )

            back_text = self.small_font.render(
                "BACK",
                True,
                (255, 255, 255)
            )

            screen.blit(
                back_text,
                (
                    back_rect.x + 22,
                    back_rect.y + 10
                )
            )

            mouse_x, mouse_y = pg.mouse.get_pos()

            if (
                pg.mouse.get_pressed()[0]
                and back_rect.collidepoint(mouse_x, mouse_y)
            ):
                state.submenu_open = None

            clip_rect = pg.Rect(
                panel.x + 20,
                panel.y + 80,
                panel.width - 40,
                panel.height - 100
            )

            old_clip = screen.get_clip()

            screen.set_clip(clip_rect)

            y = (
                panel.y +
                90 +
                state.fractal_info_scroll
            )

            for line in FRACTAL_INFORMATION_TEXT.splitlines():

                text = self.small_font.render(
                    line,
                    True,
                    (220, 220, 220)
                )

                screen.blit(
                    text,
                    (
                        panel.x + 40,
                        y
                    )
                )

                y += 34

            screen.set_clip(old_clip)

            return

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
        menu_ui
    ):

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

            pg.display.set_caption(
                f"{CONFIG.window_title} "
                f"| render {state.last_render_time_ms:.1f} ms "
                f"| {state.last_render_fps:.1f} FPS"
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
                state
            )

# ============================================================================
# APPLICATION
# ============================================================================


class Application:

    def __init__(self):

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

        self.state = create_initial_state()

        self.ui = UI()

        self.menu_ui = MenuUI()

        self.renderer = Renderer(
            self.screen
        )

    def run(self):

        while self.state.running:

            InputHandler.process_events(
                self.state
            )

            self.renderer.render_frame(
                self.state,
                self.ui,
                self.menu_ui
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