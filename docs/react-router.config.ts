import type { Config } from "@react-router/dev/config";

export default {
  ssr: false,
  basename: "/toro/",
  prerender: [
    "/",
    "/getting-started",
    "/concepts",
    "/tensor",
    "/nn",
    "/training",
    "/api-tensor",
    "/api-device",
    "/api-dtype",
    "/api-shape",
    "/api-linear",
    "/api-conv1d",
    "/api-conv2d",
    "/api-loss",
    "/api-model",
  ],
} satisfies Config;
