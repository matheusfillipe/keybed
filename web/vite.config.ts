import { defineConfig } from "vite";
import { labServer } from "./lab-server.ts";

export default defineConfig({
  plugins: [labServer()],
});
