import {
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";

const HANDEDNESS_COLORS: Record<string, string> = {
  Left: "#38bdf8",
  Right: "#f472b6",
};
const FALLBACK_COLORS = ["#38bdf8", "#f472b6"];

export function drawHands(
  ctx: CanvasRenderingContext2D,
  hands: HandLandmarkerResult,
  w: number,
  h: number,
): void {
  for (const [i, landmarks] of hands.landmarks.entries()) {
    const color =
      HANDEDNESS_COLORS[hands.handedness[i]?.[0]?.categoryName ?? ""] ??
      FALLBACK_COLORS[i % FALLBACK_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const connection of HandLandmarker.HAND_CONNECTIONS) {
      const a = landmarks[connection.start];
      const b = landmarks[connection.end];
      if (!a || !b) {
        continue;
      }
      ctx.moveTo(a.x * w, a.y * h);
      ctx.lineTo(b.x * w, b.y * h);
    }
    ctx.stroke();
    ctx.fillStyle = color;
    for (const landmark of landmarks) {
      ctx.beginPath();
      ctx.arc(landmark.x * w, landmark.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
