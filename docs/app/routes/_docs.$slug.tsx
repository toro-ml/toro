import { getDoc } from "~/content/mdx.server";
import { MdxContent } from "~/ui/mdx/mdx-components";
import { article } from "~/ui/mdx/mdx-style.css";
import { TableOfContents } from "~/ui/toc";
import type { Route } from "./+types/_docs.$slug";

export async function loader({ params }: Route.LoaderArgs) {
  const slug = params.slug;
  if (!slug) throw new Response("Not Found", { status: 404 });
  return getDoc(slug);
}

export function meta({ data }: Route.MetaArgs) {
  if (!data) return [{ title: "Not Found" }];
  return [
    { title: `${data.meta.title} | Toro` },
    { name: "description", content: data.meta.description },
  ];
}

export default function DocPage({ loaderData }: Route.ComponentProps) {
  const { meta, code, toc } = loaderData;
  return (
    <>
      <article className={article}>
        <h1>{meta.title}</h1>
        <MdxContent code={code} />
      </article>
      <TableOfContents entries={toc} />
    </>
  );
}
