import {
  BoxGeometry,
  Color,
  Mesh,
  MeshBasicMaterial,
  type PMREMGenerator,
  Scene,
  SphereGeometry,
  type Texture,
} from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

// Photometric jitter on a finished render cannot produce a specular sweep across glossy keys,
// because that highlight is a reflection of the room. These stand in for the rooms a keyboard
// gets filmed in, so the reflection moves with the camera the way a real one does.
function emitter(
  scene: Scene,
  color: number,
  intensity: number,
  size: [number, number, number],
  position: [number, number, number],
): void {
  const light = new Mesh(
    new BoxGeometry(...size),
    new MeshBasicMaterial({
      color: new Color(color).multiplyScalar(intensity),
    }),
  );
  light.position.set(...position);
  scene.add(light);
}

function shell(color: number): Scene {
  const scene = new Scene();
  const room = new Mesh(
    new BoxGeometry(60, 40, 60),
    new MeshBasicMaterial({ color, side: 1 }),
  );
  scene.add(room);
  return scene;
}

function window(): Scene {
  const scene = shell(0x0a0a0c);
  emitter(scene, 0xbfd4ff, 12, [0.5, 18, 22], [-28, 6, 0]);
  emitter(scene, 0x404246, 1, [40, 0.5, 40], [0, -14, 0]);
  return scene;
}

function overhead(): Scene {
  const scene = shell(0x14141a);
  emitter(scene, 0xffffff, 7, [26, 0.5, 14], [0, 18, 0]);
  return scene;
}

function lamp(): Scene {
  const scene = shell(0x0d0b09);
  const bulb = new Mesh(
    new SphereGeometry(3, 16, 16),
    new MeshBasicMaterial({ color: new Color(0xffc98a).multiplyScalar(16) }),
  );
  bulb.position.set(14, 12, 10);
  scene.add(bulb);
  return scene;
}

function daylight(): Scene {
  const scene = shell(0x8fb6e8);
  emitter(scene, 0xfff6e0, 9, [30, 0.5, 30], [0, 19, 0]);
  emitter(scene, 0x6b5a44, 2, [50, 0.5, 50], [0, -14, 0]);
  return scene;
}

export function makeEnvironments(pmrem: PMREMGenerator): Texture[] {
  const scenes = [
    new RoomEnvironment(),
    window(),
    overhead(),
    lamp(),
    daylight(),
  ];
  return scenes.map((scene) => pmrem.fromScene(scene as Scene, 0.04).texture);
}
