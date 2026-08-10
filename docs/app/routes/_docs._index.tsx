import { getDoc } from "~/content/mdx.server";
import { MdxContent } from "~/ui/mdx/mdx-components";
import { article } from "~/ui/mdx/mdx-style.css";
import { TableOfContents } from "~/ui/toc";
import { heroSection, heroSubtitle, heroTitle } from "./_docs._index.css";

export async function loader() {
  return getDoc("index");
}

export function meta() {
  return [
    { title: "Toro -- A lightweight ML framework for F#" },
    {
      name: "description",
      content: "PyTorch semantics, idiomatic F#. Powered by TorchSharp.",
    },
  ];
}

export default function Landing({
  loaderData,
}: {
  loaderData: Awaited<ReturnType<typeof loader>>;
}) {
  const { code, toc } = loaderData;
  return (
    <>
      <article className={article}>
        <div className={heroSection}>
          <h1 className={heroTitle}>Toro</h1>
          <div className={heroSubtitle}>
            PyTorch semantics, idiomatic F#. Powered by TorchSharp.
          </div>
        </div>
        <MdxContent code={code} />
      </article>
      <TableOfContents entries={toc} />
    </>
  );
}
