import {
  HandLandmarker,
  type HandLandmarkerResult,
  type ImageSource,
} from "@mediapipe/tasks-vision";
import { handModelUrl, type RuntimeAssets } from "./assets";

export interface HandTracker {
  detect(frame: ImageSource, timestampMs: number): HandLandmarkerResult;
}

export async function createHandTracker(
  assets: RuntimeAssets,
): Promise<HandTracker> {
  const handLandmarker = await HandLandmarker.createFromOptions(
    {
      wasmLoaderPath: assets.mediapipeLoader,
      wasmBinaryPath: assets.mediapipeWasm,
    },
    {
      baseOptions: { modelAssetPath: handModelUrl, delegate: "GPU" },
      numHands: 2,
      runningMode: "VIDEO",
    },
  );
  return {
    detect: (frame, timestampMs) =>
      handLandmarker.detectForVideo(frame, timestampMs),
  };
}
