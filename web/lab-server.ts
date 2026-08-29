import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const ROUTE = /^\/lab\/save\/([^/]+)$/;
const UNSAFE_NAME = /[^a-zA-Z0-9._-]/g;

export function labServer(): Plugin {
  const recordDir = join(
    fileURLToPath(new URL("..", import.meta.url)),
    "data",
    "recordings",
  );
  return {
    name: "kvt-lab-server",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.method !== "POST") {
          next();
          return;
        }
        const match = ROUTE.exec(req.url ?? "");
        if (!match) {
          next();
          return;
        }
        const name = match[1].replace(UNSAFE_NAME, "");
        if (!name) {
          res.statusCode = 400;
          res.end();
          return;
        }
        const chunks: Buffer[] = [];
        req.on("data", (chunk: Buffer) => {
          chunks.push(chunk);
        });
        req.on("end", () => {
          mkdir(recordDir, { recursive: true })
            .then(() => writeFile(join(recordDir, name), Buffer.concat(chunks)))
            .then(() => {
              res.statusCode = 204;
              res.end();
            })
            .catch(() => {
              res.statusCode = 500;
              res.end();
            });
        });
      });
    },
  };
}
