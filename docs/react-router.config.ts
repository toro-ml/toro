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
  ],
} satisfies Config;
