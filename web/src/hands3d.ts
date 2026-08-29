import {
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import * as THREE from "three";
import { applyHomography, type Homography } from "./homography";
import { STRIP_HEIGHT, STRIP_WIDTH } from "./strip";

const HANDEDNESS_COLORS: Record<string, string> = {
  Left: "#38bdf8",
  Right: "#f472b6",
};
const FALLBACK_COLORS = ["#38bdf8", "#f472b6"];
const KEYBOARD_WIDTH = 52;
const KEYBOARD_DEPTH = 6.5;
const HEIGHT_CLAMP = 0.15;
const HEIGHT_SCALE = 20;
const DOT_SIZE = 0.28;
const MAX_HANDS = 2;
const LANDMARK_COUNT = 21;

interface HandSlot {
  lines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  points: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>;
  linePositions: Float32Array;
  pointPositions: Float32Array;
  lineAttribute: THREE.BufferAttribute;
  pointAttribute: THREE.BufferAttribute;
}

export interface Hands3D {
  update(
    hands: HandLandmarkerResult,
    toStrip: Homography,
    width: number,
    height: number,
  ): void;
}

export function createHands3D(parent: THREE.Object3D): Hands3D {
  const slots: HandSlot[] = [];
  for (let i = 0; i < MAX_HANDS; i += 1) {
    const linePositions = new Float32Array(LANDMARK_COUNT * 2 * 3);
    const pointPositions = new Float32Array(LANDMARK_COUNT * 3);
    const lineGeometry = new THREE.BufferGeometry();
    const lineAttribute = new THREE.BufferAttribute(linePositions, 3);
    lineGeometry.setAttribute("position", lineAttribute);
    const pointGeometry = new THREE.BufferGeometry();
    const pointAttribute = new THREE.BufferAttribute(pointPositions, 3);
    pointGeometry.setAttribute("position", pointAttribute);
    const lines = new THREE.LineSegments(
      lineGeometry,
      new THREE.LineBasicMaterial(),
    );
    const points = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({ size: DOT_SIZE }),
    );
    lines.frustumCulled = false;
    points.frustumCulled = false;
    lines.visible = false;
    points.visible = false;
    parent.add(lines, points);
    slots.push({
      lines,
      points,
      linePositions,
      pointPositions,
      lineAttribute,
      pointAttribute,
    });
  }

  return {
    update: (hands, toStrip, width, height) => {
      for (const [slotIndex, slot] of slots.entries()) {
        const landmarks = hands.landmarks[slotIndex];
        if (!landmarks) {
          slot.lines.visible = false;
          slot.points.visible = false;
          continue;
        }
        const world = hands.worldLandmarks[slotIndex];
        for (const [i, landmark] of landmarks.entries()) {
          if (i >= LANDMARK_COUNT) {
            break;
          }
          const strip = applyHomography(
            toStrip,
            landmark.x * width,
            landmark.y * height,
          );
          const u = strip.x / STRIP_WIDTH;
          const v = strip.y / STRIP_HEIGHT;
          slot.pointPositions[i * 3] = u * KEYBOARD_WIDTH - KEYBOARD_WIDTH / 2;
          const worldZ = world?.[i]?.z ?? 0;
          slot.pointPositions[i * 3 + 1] =
            Math.min(HEIGHT_CLAMP, Math.max(-HEIGHT_CLAMP, -worldZ)) *
            HEIGHT_SCALE;
          slot.pointPositions[i * 3 + 2] =
            v * KEYBOARD_DEPTH - KEYBOARD_DEPTH / 2;
        }
        for (const [
          i,
          connection,
        ] of HandLandmarker.HAND_CONNECTIONS.entries()) {
          const a = connection.start * 3;
          const b = connection.end * 3;
          slot.linePositions[i * 6] = slot.pointPositions[a];
          slot.linePositions[i * 6 + 1] = slot.pointPositions[a + 1];
          slot.linePositions[i * 6 + 2] = slot.pointPositions[a + 2];
          slot.linePositions[i * 6 + 3] = slot.pointPositions[b];
          slot.linePositions[i * 6 + 4] = slot.pointPositions[b + 1];
          slot.linePositions[i * 6 + 5] = slot.pointPositions[b + 2];
        }
        const categoryName =
          hands.handedness[slotIndex]?.[0]?.categoryName ?? "";
        const color =
          HANDEDNESS_COLORS[categoryName] ??
          FALLBACK_COLORS[slotIndex % FALLBACK_COLORS.length];
        slot.lines.material.color.set(color);
        slot.points.material.color.set(color);
        slot.lineAttribute.needsUpdate = true;
        slot.pointAttribute.needsUpdate = true;
        slot.lines.visible = true;
        slot.points.visible = true;
      }
    },
  };
}
