import { compile } from "@mdx-js/mdx";
import rehypeShiki from "@shikijs/rehype";
import fs from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { Root, Element } from "hast";
import type { Plugin } from "unified";

const contentDir = path.join(process.cwd(), "app", "content");

export interface TocEntry {
  id: string;
  text: string;
  depth: number;
}

export interface DocMeta {
  slug: string;
  title: string;
  description: string;
  namespace?: string;
  showTitle: boolean;
}

function extractText(node: Element): string {
  let text = "";
  for (const child of node.children) {
    if (child.type === "text") text += child.value;
    else if (child.type === "element") text += extractText(child);
  }
  return text;
}

function rehypeExtractToc(toc: TocEntry[]): Plugin<[], Root> {
  return () => (tree) => {
    for (const node of tree.children) {
      if (node.type !== "element") continue;
      const m = node.tagName.match(/^h([1-4])$/);
      if (!m) continue;
      const id = (node.properties?.id as string) ?? "";
      const text = extractText(node);
      if (id && text) toc.push({ id, text, depth: Number(m[1]) });
    }
  };
}

async function resolveSlug(slug: string): Promise<string> {
  const direct = path.join(contentDir, `${slug}.mdx`);
  try {
    await fs.access(direct);
    return direct;
  } catch {
    // search subdirectories
  }
  const entries = await fs.readdir(contentDir, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const nested = path.join(contentDir, entry.name, `${slug}.mdx`);
    try {
      await fs.access(nested);
      return nested;
    } catch {
      continue;
    }
  }
  throw new Error(`MDX not found: ${slug}`);
}

export async function getDoc(slug: string) {
  const filePath = await resolveSlug(slug);
  const raw = await fs.readFile(filePath, "utf-8");
  const { data, content } = matter(raw);

  const toc: TocEntry[] = [];

  const compiled = await compile(content, {
    outputFormat: "function-body",
    remarkPlugins: [remarkGfm, remarkMath],
    rehypePlugins: [
      rehypeSlug,
      rehypeKatex,
      [
        rehypeShiki,
        {
          theme: "github-dark-default",
          defaultColor: false,
          langs: ["fsharp", "bash", "typescript", "json", "xml", "csharp"],
        },
      ],
      rehypeExtractToc(toc),
    ],
  });

  const title = data.title as string;
  const showTitle = (data.showTitle as boolean | undefined) ?? true;
  if (showTitle) {
    toc.unshift({ id: "top", text: title, depth: 1 });
  }

  return {
    meta: {
      slug,
      title,
      description: (data.description as string) ?? "",
      namespace: (data.namespace as string) ?? undefined,
      showTitle,
    },
    code: String(compiled),
    toc,
  };
}

export async function getAllSlugs(): Promise<string[]> {
  const slugs: string[] = [];
  const entries = await fs.readdir(contentDir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isFile() && entry.name.endsWith(".mdx")) {
      slugs.push(entry.name.replace(/\.mdx$/, ""));
    } else if (entry.isDirectory()) {
      const nested = await fs.readdir(path.join(contentDir, entry.name));
      for (const f of nested) {
        if (f.endsWith(".mdx")) slugs.push(f.replace(/\.mdx$/, ""));
      }
    }
  }
  return slugs;
}
