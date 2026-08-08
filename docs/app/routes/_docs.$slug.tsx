import { useEffect } from "react";
import { Link, useLocation } from "react-router";
import { getDoc } from "~/content/mdx.server";
import { MdxContent } from "~/ui/mdx/mdx-components";
import {
  article,
  namespaceBadge,
  pageNav,
  pageNavLabel,
  pageNavLink,
  pageNavNext,
  pageNavTitle,
} from "~/ui/mdx/mdx-style.css";
import { navItems } from "~/ui/sidebar";
import { TableOfContents } from "~/ui/toc";
import type { Route } from "./+types/_docs.$slug";

function setTransitionDirection(dir: "forward" | "back") {
  document.documentElement.dataset.direction = dir;
}

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
  const currentIndex = navItems.findIndex(
    (item) => item.to === `/${params.slug}`,
  );
  const prev = currentIndex > 0 ? navItems[currentIndex - 1] : null;
  const next =
    currentIndex < navItems.length - 1 ? navItems[currentIndex + 1] : null;

  useEffect(() => {
    delete document.documentElement.dataset.direction;
  }, [location.pathname]);

  return (
    <>
      <article className={article}>
        {meta.namespace && (
          <div className={namespaceBadge}>
            <code>{meta.namespace}</code>
          </div>
        )}
        {meta.showTitle && <h1 id="top">{meta.title}</h1>}
        <MdxContent code={code} />
        {(prev || next) && (
          <nav className={pageNav}>
            {prev ? (
              <Link
                to={prev.to}
                className={pageNavLink}
                viewTransition
                onClick={() => setTransitionDirection("back")}
              >
                <span className={pageNavLabel}>Previous</span>
                <span className={pageNavTitle}>{prev.label}</span>
              </Link>
            ) : (
              <span />
            )}
            {next && (
              <Link
                to={next.to}
                className={pageNavNext}
                viewTransition
                onClick={() => setTransitionDirection("forward")}
              >
                <span className={pageNavLabel}>Next</span>
                <span className={pageNavTitle}>{next.label}</span>
              </Link>
            )}
          </nav>
        )}
      </article>
      <TableOfContents entries={toc} />
    </>
  );
}
