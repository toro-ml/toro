import type { Config } from "@react-router/dev/config";
import { readdirSync } from "node:fs";
import { join } from "node:path";

const contentDir = join(import.meta.dirname, "app", "content");
const slugs = readdirSync(contentDir)
  .filter((f) => f.endsWith(".mdx"))
  .map((f) => `/${f.replace(/\.mdx$/, "")}`);

export default {
  ssr: false,
  basename: "/toro/",
  prerender: ["/", ...slugs],
} satisfies Config;
