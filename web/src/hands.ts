import {
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import wasmLoaderUrl from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import wasmBinaryUrl from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export interface HandTracker {
  detect(video: HTMLVideoElement, timestampMs: number): HandLandmarkerResult;
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
    detect: (video, timestampMs) =>
      handLandmarker.detectForVideo(video, timestampMs),
  };
}
