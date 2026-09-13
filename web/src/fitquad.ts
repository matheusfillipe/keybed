import type { Point } from "./homography";

const MIN_AREA_PX = 60;
const MIN_SIDE_POINTS = 8;
const TRIM_FRACTION = 0.12;
const EPSILON_STEPS = 30;
const EPSILON_LOW = 0.001;
const EPSILON_HIGH = 0.2;

// the mask's largest blob is the keybed; specks elsewhere are not worth fitting a quad to
function largestBlobBoundary(
  mask: Uint8Array,
  width: number,
  height: number,
): Point[] | null {
  const labels = new Int32Array(width * height).fill(-1);
  const stack: number[] = [];
  let best: number[] = [];
  let bestSize = 0;

  for (let start = 0; start < mask.length; start += 1) {
    if (mask[start] === 0 || labels[start] !== -1) {
      continue;
    }
    const blob: number[] = [];
    labels[start] = start;
    stack.push(start);
    while (stack.length > 0) {
      const index = stack.pop() as number;
      blob.push(index);
      const x = index % width;
      const y = (index - x) / width;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) {
            continue;
          }
          const next = ny * width + nx;
          if (mask[next] !== 0 && labels[next] === -1) {
            labels[next] = start;
            stack.push(next);
          }
        }
      }
    }
    if (blob.length > bestSize) {
      bestSize = blob.length;
      best = blob;
    }
  }
  if (bestSize < MIN_AREA_PX) {
    return null;
  }

  const boundary: Point[] = [];
  for (const index of best) {
    const x = index % width;
    const y = (index - x) / width;
    const edge =
      x === 0 ||
      y === 0 ||
      x === width - 1 ||
      y === height - 1 ||
      mask[index - 1] === 0 ||
      mask[index + 1] === 0 ||
      mask[index - width] === 0 ||
      mask[index + width] === 0;
    if (edge) {
      boundary.push({ x, y });
    }
  }
  return boundary.length >= 4 ? boundary : null;
}

function cross(o: Point, a: Point, b: Point): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function convexHull(points: Point[]): Point[] {
  const sorted = [...points].sort((p, q) => p.x - q.x || p.y - q.y);
  const build = (input: Point[]): Point[] => {
    const out: Point[] = [];
    for (const point of input) {
      while (
        out.length >= 2 &&
        cross(out[out.length - 2], out[out.length - 1], point) <= 0
      ) {
        out.pop();
      }
      out.push(point);
    }
    out.pop();
    return out;
  };
  return [...build(sorted), ...build([...sorted].reverse())];
}

function segmentDistance(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq < 1e-9) {
    return Math.hypot(p.x - a.x, p.y - a.y);
  }
  const t = Math.max(
    0,
    Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq),
  );
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

// a hull is a closed ring, so anchoring on index 0 and n-1 would pin a corner at the leftmost
// point instead of a real one; splitting at the farthest point gives two honest open chains
function simplifyClosed(polygon: Point[], epsilon: number): Point[] {
  const count = polygon.length;
  if (count <= 4) {
    return polygon;
  }
  let split = 0;
  let farthest = -1;
  for (let i = 1; i < count; i += 1) {
    const distance = Math.hypot(
      polygon[i].x - polygon[0].x,
      polygon[i].y - polygon[0].y,
    );
    if (distance > farthest) {
      farthest = distance;
      split = i;
    }
  }
  const head = simplify(polygon.slice(0, split + 1), epsilon);
  const tail = simplify([...polygon.slice(split), polygon[0]], epsilon);
  return [...head.slice(0, -1), ...tail.slice(0, -1)];
}

function simplify(polygon: Point[], epsilon: number): Point[] {
  if (polygon.length < 3) {
    return polygon;
  }
  const keep = new Array<boolean>(polygon.length).fill(false);
  keep[0] = true;
  keep[polygon.length - 1] = true;
  const stack: [number, number][] = [[0, polygon.length - 1]];
  while (stack.length > 0) {
    const [first, last] = stack.pop() as [number, number];
    let worst = 0;
    let index = -1;
    for (let i = first + 1; i < last; i += 1) {
      const distance = segmentDistance(
        polygon[i],
        polygon[first],
        polygon[last],
      );
      if (distance > worst) {
        worst = distance;
        index = i;
      }
    }
    if (index !== -1 && worst > epsilon) {
      keep[index] = true;
      stack.push([first, index], [index, last]);
    }
  }
  return polygon.filter((_, i) => keep[i]);
}

function perimeter(polygon: Point[]): number {
  let total = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    total += Math.hypot(b.x - a.x, b.y - a.y);
  }
  return total;
}

// the loosest simplification that still keeps four sides is the one that found the real corners
function seedQuad(hull: Point[]): Point[] | null {
  const length = perimeter(hull);
  if (length <= 0) {
    return null;
  }
  let low = EPSILON_LOW;
  let high = EPSILON_HIGH;
  for (let i = 0; i < EPSILON_STEPS; i += 1) {
    const middle = (low + high) / 2;
    if (simplifyClosed(hull, middle * length).length > 4) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const approx = simplifyClosed(hull, high * length);
  return approx.length === 4 ? approx : null;
}

interface Line {
  point: Point;
  dx: number;
  dy: number;
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
  // principal axis of a 2x2 covariance, closed form rather than a full svd
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

function trimEnds(side: Point[], a: Point, b: Point): Point[] {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = dx * dx + dy * dy;
  if (lengthSq < 1e-9) {
    return side;
  }
  const kept = side.filter((p) => {
    const t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq;
    return t > TRIM_FRACTION && t < 1 - TRIM_FRACTION;
  });
  return kept.length >= MIN_SIDE_POINTS ? kept : side;
}

export function quadFromMask(
  mask: Uint8Array,
  width: number,
  height: number,
): Point[] | null {
  const boundary = largestBlobBoundary(mask, width, height);
  if (!boundary) {
    return null;
  }
  const hull = convexHull(boundary);
  if (hull.length < 4) {
    return null;
  }
  const seed = seedQuad(hull);
  if (!seed) {
    return null;
  }
  const sides: Point[][] = [[], [], [], []];
  for (const point of boundary) {
    let bestIndex = 0;
    let best = Infinity;
    for (let i = 0; i < 4; i += 1) {
      const distance = segmentDistance(point, seed[i], seed[(i + 1) % 4]);
      if (distance < best) {
        best = distance;
        bestIndex = i;
      }
    }
    sides[bestIndex].push(point);
  }
  const lines: Line[] = [];
  for (let i = 0; i < 4; i += 1) {
    if (sides[i].length < MIN_SIDE_POINTS) {
      return seed;
    }
    lines.push(fitLine(trimEnds(sides[i], seed[i], seed[(i + 1) % 4])));
  }
  const corners: Point[] = [];
  for (let i = 0; i < 4; i += 1) {
    const corner = intersect(lines[(i + 3) % 4], lines[i]);
    if (!corner) {
      return seed;
    }
    corners.push(corner);
  }
  return corners;
}
