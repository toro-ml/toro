import type { Config } from "@react-router/dev/config";
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const contentDir = join(import.meta.dirname, "app", "content");

function collectSlugs(dir: string): string[] {
  const slugs: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isFile() && entry.name.endsWith(".mdx")) {
      slugs.push(`/${entry.name.replace(/\.mdx$/, "")}`);
    } else if (entry.isDirectory()) {
      slugs.push(...collectSlugs(join(dir, entry.name)));
    }
  }
  return slugs;
}

export default {
  ssr: false,
  basename: "/toro/",
  prerender: ["/", ...collectSlugs(contentDir)],
} satisfies Config;
