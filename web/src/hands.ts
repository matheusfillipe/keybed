import {
  HandLandmarker,
  type HandLandmarkerResult,
  type ImageSource,
} from "@mediapipe/tasks-vision";
import wasmLoaderUrl from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import wasmBinaryUrl from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export interface HandTracker {
  detect(frame: ImageSource, timestampMs: number): HandLandmarkerResult;
}

export async function createHandTracker(): Promise<HandTracker> {
  const handLandmarker = await HandLandmarker.createFromOptions(
    { wasmLoaderPath: wasmLoaderUrl, wasmBinaryPath: wasmBinaryUrl },
    {
      baseOptions: { modelAssetPath: MODEL_URL },
      numHands: 2,
      runningMode: "VIDEO",
    },
  );
  return {
    detect: (frame, timestampMs) =>
      handLandmarker.detectForVideo(frame, timestampMs),
  };
}
