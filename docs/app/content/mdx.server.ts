import { compile } from "@mdx-js/mdx";
import rehypeShiki from "@shikijs/rehype";
import fs from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
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
      const m = node.tagName.match(/^h([23])$/);
      if (!m) continue;
      const id = (node.properties?.id as string) ?? "";
      const text = extractText(node);
      if (id && text) toc.push({ id, text, depth: Number(m[1]) });
    }
  };
}

export async function getDoc(slug: string) {
  const filePath = path.join(contentDir, `${slug}.mdx`);
  const raw = await fs.readFile(filePath, "utf-8");
  const { data, content } = matter(raw);

  const toc: TocEntry[] = [];

  const compiled = await compile(content, {
    outputFormat: "function-body",
    remarkPlugins: [remarkGfm],
    rehypePlugins: [
      rehypeSlug,
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

  return {
    meta: {
      slug,
      title: data.title as string,
      description: (data.description as string) ?? "",
    },
    code: String(compiled),
    toc,
  };
}

export async function getAllSlugs(): Promise<string[]> {
  const files = await fs.readdir(contentDir);
  return files
    .filter((f) => f.endsWith(".mdx"))
    .map((f) => f.replace(/\.mdx$/, ""));
}
