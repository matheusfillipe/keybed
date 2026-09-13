import type { Point } from "./homography";

const SAMPLES = 48;
// measured over the corpus on 640 wide frames: 10 px lands on the true edge, wider searches
// find the wrong one (key separators, the case, the floor line)
export const SEARCH_PX = 10;
export const SEARCH_STEP = 0.5;
export const MIN_RESPONSE = 6;
const MIN_POINTS = 8;
const TRIM = 0.08;
const BLUR_SIGMA = 1.2;
// refinement nudges an edge onto a gradient; a corner that moves further than the search
// window did not find its edge, it found somebody else's, so the coarse quad is kept
const MAX_SHIFT = 4;

let blurred = new Float32Array(0);
let scratch = new Float32Array(0);

// without this the gradient search locks onto individual key separators instead of the keybed edge
export function blur(
  gray: Float32Array,
  width: number,
  height: number,
): Float32Array {
  const radius = Math.ceil(3 * BLUR_SIGMA);
  const kernel = new Float32Array(radius * 2 + 1);
  let total = 0;
  for (let i = -radius; i <= radius; i += 1) {
    const value = Math.exp(-(i * i) / (2 * BLUR_SIGMA * BLUR_SIGMA));
    kernel[i + radius] = value;
    total += value;
  }
  for (let i = 0; i < kernel.length; i += 1) {
    kernel[i] /= total;
  }
  if (blurred.length !== gray.length) {
    blurred = new Float32Array(gray.length);
    scratch = new Float32Array(gray.length);
  }
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const sx = Math.min(width - 1, Math.max(0, x + k));
        sum += gray[row + sx] * kernel[k + radius];
      }
      scratch[row + x] = sum;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const sy = Math.min(height - 1, Math.max(0, y + k));
        sum += scratch[sy * width + x] * kernel[k + radius];
      }
      blurred[y * width + x] = sum;
    }
  }
  return blurred;
}

interface Line {
  point: Point;
  dx: number;
  dy: number;
}

export function bilinear(
  gray: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
): number {
  const cx = Math.max(0, Math.min(width - 1.001, x));
  const cy = Math.max(0, Math.min(height - 1.001, y));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const fx = cx - x0;
  const fy = cy - y0;
  const row = y0 * width + x0;
  const a = gray[row];
  const b = gray[row + 1];
  const c = gray[row + width];
  const d = gray[row + width + 1];
  return (
    a * (1 - fx) * (1 - fy) +
    b * fx * (1 - fy) +
    c * (1 - fx) * fy +
    d * fx * fy
  );
}

function fitLine(points: Point[]): Line {
  let mx = 0;
  let my = 0;
  for (const p of points) {
    mx += p.x;
    my += p.y;
  }
  mx /= points.length;
  my /= points.length;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const p of points) {
    const dx = p.x - mx;
    const dy = p.y - my;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return { point: { x: mx, y: my }, dx: Math.cos(theta), dy: Math.sin(theta) };
}

function intersect(a: Line, b: Line): Point | null {
  const determinant = a.dx * -b.dy - -b.dx * a.dy;
  if (Math.abs(determinant) < 1e-9) {
    return null;
  }
  const rx = b.point.x - a.point.x;
  const ry = b.point.y - a.point.y;
  const t = (rx * -b.dy - -b.dx * ry) / determinant;
  return { x: a.point.x + t * a.dx, y: a.point.y + t * a.dy };
}

function refineEdge(
  gray: Float32Array,
  width: number,
  height: number,
  start: Point,
  end: Point,
  searchPx: number,
): Point[] | null {
  const ex = end.x - start.x;
  const ey = end.y - start.y;
  const length = Math.hypot(ex, ey);
  if (length < 1) {
    return null;
  }
  const nx = -ey / length;
  const ny = ex / length;
  const found: Point[] = [];
  for (let s = 0; s < SAMPLES; s += 1) {
    const t = TRIM + ((1 - 2 * TRIM) * s) / (SAMPLES - 1);
    const bx = start.x + t * ex;
    const by = start.y + t * ey;
    let bestResponse = 0;
    let bestOffset = 0;
    // the edge sits where brightness changes fastest along the normal, not where the mask stopped
    for (let o = -searchPx; o <= searchPx; o += SEARCH_STEP) {
      const before = bilinear(
        gray,
        width,
        height,
        bx + (o - SEARCH_STEP) * nx,
        by + (o - SEARCH_STEP) * ny,
      );
      const after = bilinear(
        gray,
        width,
        height,
        bx + (o + SEARCH_STEP) * nx,
        by + (o + SEARCH_STEP) * ny,
      );
      const response = Math.abs(after - before) / (2 * SEARCH_STEP);
      if (response > bestResponse) {
        bestResponse = response;
        bestOffset = o;
      }
    }
    if (bestResponse > MIN_RESPONSE) {
      found.push({ x: bx + bestOffset * nx, y: by + bestOffset * ny });
    }
  }
  return found.length >= MIN_POINTS ? found : null;
}

export function refineQuad(
  gray: Float32Array,
  width: number,
  height: number,
  quad: Point[],
  searchPx: number = SEARCH_PX,
): Point[] {
  const smooth = blur(gray, width, height);
  const lines: Line[] = [];
  for (let i = 0; i < 4; i += 1) {
    const points = refineEdge(
      smooth,
      width,
      height,
      quad[i],
      quad[(i + 1) % 4],
      searchPx,
    );
    if (!points) {
      return quad;
    }
    lines.push(fitLine(points));
  }
  const corners: Point[] = [];
  for (let i = 0; i < 4; i += 1) {
    const corner = intersect(lines[(i + 3) % 4], lines[i]);
    if (!corner) {
      return quad;
    }
    corners.push(corner);
  }
  for (let i = 0; i < 4; i += 1) {
    if (
      Math.hypot(corners[i].x - quad[i].x, corners[i].y - quad[i].y) >
      MAX_SHIFT * searchPx
    ) {
      return quad;
    }
  }
  return corners;
}
