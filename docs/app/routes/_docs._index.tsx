import { Flex, Heading, Text } from "@radix-ui/themes";
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
      content: "A lightweight ML framework for F# powered by TorchSharp.",
    },
  ];
}

export default function Landing({ loaderData }: { loaderData: Awaited<ReturnType<typeof loader>> }) {
  const { code, toc } = loaderData;
  return (
    <>
      <article className={article}>
        <Flex direction="column" gap="1" className={heroSection}>
          <Heading size="8" className={heroTitle}>
            Toro
          </Heading>
          <Text size="3" className={heroSubtitle}>
            A lightweight ML framework for F# powered by TorchSharp.
          </Text>
        </Flex>
        <MdxContent code={code} />
      </article>
      <TableOfContents entries={toc} />
    </>
  );
}
