import type { Detection } from "./detector";
import type { Point } from "./homography";
/** A keybed the picture supports, or the reason it was refused. Nothing on
 * screen beats the most likely wrong thing, so a refusal draws nothing. */
export type Lock = {
    readonly held: true;
    /** Corners in frame coordinates, 0 to 1, ordered so 0 to 1 runs along the
     * back edge and 3 to 2 along the edge the player stands at. */
    readonly quad: Point[];
    readonly inputQuad: Point[];
    readonly margin: number;
} | {
    readonly held: false;
    readonly reason: string;
};
export declare const minConfidence = 0.6;
export declare const maxPoseResidual = 1;
/** Everything the picture has to say before a keybed is drawn: the mask filled
 * it, the shape is a keybed, it stands as a rectangle in 3D, and the black keys
 * are there. */
export declare function lockKeybed(detection: Detection, frame: {
    width: number;
    height: number;
}): Lock;
