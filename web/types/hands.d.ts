import { type HandLandmarkerResult, type ImageSource } from "@mediapipe/tasks-vision";
import { type RuntimeAssets } from "./assets";
export interface HandTracker {
    detect(frame: ImageSource, timestampMs: number): HandLandmarkerResult;
}
export declare function createHandTracker(assets: RuntimeAssets): Promise<HandTracker>;
