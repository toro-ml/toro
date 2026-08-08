import { useEffect, useState } from "react";
import type { TocEntry } from "~/content/mdx.server";
import * as s from "./toc-style.css";

function useActiveHeading(ids: string[]) {
  const [activeId, setActiveId] = useState("");

  useEffect(() => {
    if (ids.length === 0) return;

    const onScroll = () => {
      let current = "";
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= window.innerHeight * 0.2) {
          current = id;
        }
      }

      const atBottom =
        window.scrollY + window.innerHeight >=
        document.documentElement.scrollHeight - 1;
      if (atBottom) {
        for (let i = ids.length - 1; i >= 0; i--) {
          const el = document.getElementById(ids[i]);
          if (el && el.getBoundingClientRect().top < window.innerHeight) {
            current = ids[i];
            break;
          }
        }
      }

      setActiveId(current);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [ids]);

  return activeId;
}

export const TableOfContents = ({ entries }: { entries: TocEntry[] }) => {
  const activeId = useActiveHeading(entries.map((e) => e.id));

  if (entries.length === 0) return null;
  return (
    <nav className={s.nav} aria-label="Table of contents">
      <div className={s.heading}>On this page</div>
      <ul className={s.list}>
        {entries.map(({ id, text, depth }) => (
          <li key={id}>
            <a
              href={`#${id}`}
              className={[
                s.link,
                depth >= 3 ? s.linkDepth3 : "",
                activeId === id ? s.linkActive : "",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              {text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
};
