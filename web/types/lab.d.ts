import type { Corners } from "./calibrate";
export interface LabOptions {
    video: HTMLVideoElement;
    stream: MediaStream;
    mount: HTMLElement;
    getCorners(): Corners | null;
}
export declare function createLab(options: LabOptions): void;
