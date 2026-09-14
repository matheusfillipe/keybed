import wasmLoaderUrl from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import wasmBinaryUrl from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url";
import type { RuntimeAssets } from "./assets";

export const viteAssets: RuntimeAssets = {
  ortWasm: ortWasmUrl,
  mediapipeLoader: wasmLoaderUrl,
  mediapipeWasm: wasmBinaryUrl,
};
