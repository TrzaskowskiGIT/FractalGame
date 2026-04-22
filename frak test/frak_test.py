import cupy as cp
import pygame as pg
import numpy as np
import time

UPSCALE = 1

WIDTH, HEIGHT = (int)(1920 * UPSCALE), (int)(1080 * UPSCALE)

gpu_output = cp.zeros(WIDTH * HEIGHT, dtype=cp.int32)

mandelbrot_kernel = cp.RawKernel(r'''
struct dd {
    double hi;
    double lo;
};

__device__ dd dd_add(dd a, dd b)
{
    double s = a.hi + b.hi;
    double v = s - a.hi;
    double t = ((b.hi - v) + (a.hi - (s - v))) + a.lo + b.lo;

    dd r;
    r.hi = s + t;
    r.lo = t - (r.hi - s);
    return r;
}

__device__ dd dd_mul(dd a, dd b)
{
    double p = a.hi * b.hi;

    double err = fma(a.hi, b.hi, -p);
    err += a.hi*b.lo;
    err += a.lo*b.hi;

    dd r;
    r.hi = p + err;
    r.lo = err - (r.hi - p);
    return r;
}

__device__ double dd_to_double(dd a)
{
    return a.hi + a.lo;
}

extern "C" __global__
void mandelbrot(
    double xmin, double xmax,
    double ymin, double ymax,
    int width, int height,
    int max_iter,
    int *output)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= width || y >= height) return;

    dd c_r;
    c_r.hi = xmin + x * (xmax - xmin) / width;
    c_r.lo = 0.0;

    dd c_i;
    c_i.hi = ymax - y * (ymax - ymin) / height;
    c_i.lo = 0.0;

    dd zr = {0.0,0.0};
    dd zi = {0.0,0.0};


    int i;

    for(i=0;i<max_iter;i++)
    {
        dd zr2 = dd_mul(zr,zr);
        dd zi2 = dd_mul(zi,zi);

        dd temp1;
        temp1.hi = zr2.hi - zi2.hi;
        temp1.lo = zr2.lo - zi2.lo;
        dd zr_new = dd_add(temp1, c_r);

        dd zrzi = dd_mul(zr,zi);
        dd temp2;
        temp2.hi = 2.0 * zrzi.hi;
        temp2.lo = 2.0 * zrzi.lo;
        dd zi_new = dd_add(temp2, c_i);

        zr = zr_new;
        zi = zi_new;

        dd zr2m = dd_mul(zr, zr);
        dd zi2m = dd_mul(zi, zi);

        double mag = zr2m.hi + zi2m.hi;

        if (mag > 4.0)
            break;

        if(mag > 4.0)
            break;
    }

    output[y*width + x] = i;
}
''','mandelbrot')





def init_pygame():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    return screen, clock


def compute_mandelbrot(xmin, xmax, ymin, ymax, max_iter):

    output = gpu_output

    block = (16, 16)
    grid = (
        (WIDTH + block[0] - 1) // block[0],
        (HEIGHT + block[1] - 1) // block[1]
    )

    mandelbrot_kernel(
        grid,
        block,
        (
            np.float64(xmin),
            np.float64(xmax),
            np.float64(ymin),
            np.float64(ymax),
            np.int32(WIDTH),
            np.int32(HEIGHT),
            np.int32(max_iter),
            output
        )
    )

    mset = output.reshape((HEIGHT, WIDTH))

    img = (mset * 255 / max_iter).astype(cp.uint8)

    t = cp.transpose(img) / 255

    r = (9*(1-t)*t*t*t*255).astype(cp.uint8)
    g = (15*(1-t)*(1-t)*t*t*255).astype(cp.uint8)
    b = (8.5*(1-t)*(1-t)*(1-t)*t*255).astype(cp.uint8)

    rgb = cp.stack([r, g, b], axis=2)

    surface = pg.surfarray.make_surface(cp.asnumpy(rgb))

    return surface


def center_camera(mx, my, xmin, xmax, ymin, ymax):
    cx = xmin + (mx / WIDTH) * (xmax - xmin)
    cy = ymax - (my / HEIGHT) * (ymax - ymin)

    xr = xmax - xmin
    yr = ymax - ymin

    xmin = cx - xr / 2
    xmax = cx + xr / 2
    ymin = cy - yr / 2
    ymax = cy + yr / 2

    return xmin, xmax, ymin, ymax


def zoom_camera(button, xmin, xmax, ymin, ymax):
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    zoom = 0.8 if button == 4 else 1.25

    xr = (xmax - xmin) * zoom
    yr = (ymax - ymin) * zoom

    xmin = cx - xr / 2
    xmax = cx + xr / 2
    ymin = cy - yr / 2
    ymax = cy + yr / 2

    return xmin, xmax, ymin, ymax


def handle_events(xmin, xmax, ymin, ymax, max_iter):
    needs_update = False
    running = True

    for event in pg.event.get():

        if event.type == pg.QUIT:
            running = False

        if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
            running = False

        if event.type == pg.KEYDOWN and event.key == pg.K_w:
            max_iter = max_iter * 2
            needs_update = True

        if event.type == pg.KEYDOWN and event.key == pg.K_s:
            max_iter = (int)(max_iter / 2)
            needs_update = True

        if event.type == pg.MOUSEBUTTONDOWN:

            if event.button == 1:
                mx, my = pg.mouse.get_pos()
                xmin, xmax, ymin, ymax = center_camera(mx, my, xmin, xmax, ymin, ymax)
                needs_update = True

            elif event.button in (4, 5):
                xmin, xmax, ymin, ymax = zoom_camera(event.button, xmin, xmax, ymin, ymax)
                needs_update = True

    return running, needs_update, xmin, xmax, ymin, ymax, max_iter


def main():
    screen, clock = init_pygame()

    xmin, xmax = -3.0, 1.5
    x_range = xmax - xmin

    aspect_ratio = HEIGHT / WIDTH
    y_range = x_range * aspect_ratio

    ymin = -y_range / 2
    ymax = y_range / 2

    needs_update = True
    surface = None
    running = True
    max_iter = 100

    while running:

        running, event_update, xmin, xmax, ymin, ymax, max_iter = handle_events(xmin, xmax, ymin, ymax, max_iter)

        needs_update |= event_update

        if needs_update:
            start_time = time.time()
            surface = compute_mandelbrot(xmin, xmax, ymin, ymax, max_iter)
            end_time = time.time()
            print("FPS:",1 / (end_time - start_time))
            print("Itrations: ", max_iter)
            needs_update = False

        if surface is not None:
            screen.blit(surface, (0, 0))


        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()