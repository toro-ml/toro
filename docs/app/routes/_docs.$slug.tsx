import { useEffect } from "react";
import { useLocation } from "react-router";
import { getDoc } from "~/content/mdx.server";
import { MdxContent } from "~/ui/mdx/mdx-components";
import { article } from "~/ui/mdx/mdx-style.css";
import { adjacentNavItems } from "~/ui/nav";
import { PageNav } from "~/ui/page-nav";
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

export default function DocPage({
  loaderData,
  params,
}: Route.ComponentProps) {
  const { meta, code, toc } = loaderData;
  const location = useLocation();
  const { previous, next } = adjacentNavItems(`/${params.slug}`);

  useEffect(() => {
    delete document.documentElement.dataset.direction;
  }, [location.pathname]);

  return (
    <>
      <article className={article}>
        {meta.showTitle && <h1 id="top">{meta.title}</h1>}
        <MdxContent code={code} />
        <PageNav previous={previous} next={next} />
      </article>
      <TableOfContents entries={toc} />
    </>
  );
}
