import { useEffect, useMemo, useState } from "react";
import type { TocEntry } from "~/content/mdx.server";
import { classNames } from "~/ui/class-names";
import * as s from "./toc-style.css";

interface HeadingPosition {
  id: string;
  top: number;
}

function readHeadingPositions(ids: readonly string[]): HeadingPosition[] {
  return ids
    .map((id) => {
      const element = document.getElementById(id);
      return element
        ? { id, top: element.getBoundingClientRect().top }
        : null;
    })
    .filter((position): position is HeadingPosition => position !== null);
}

function resolveActiveHeading(
  positions: readonly HeadingPosition[],
  viewportHeight: number,
  atBottom: boolean,
) {
  const current = positions.reduce(
    (activeId, { id, top }) =>
      top <= viewportHeight * 0.2 ? id : activeId,
    "",
  );

  if (!atBottom) return current;

  return (
    positions.reduceRight(
      (activeId, { id, top }) =>
        activeId || (top < viewportHeight ? id : ""),
      "",
    ) || current
  );
}

function depthClass(depth: number) {
  if (depth >= 4) return s.linkDepth4;
  return {
    2: s.linkDepth2,
    3: s.linkDepth3,
  }[depth] ?? "";
}

function useActiveHeading(ids: readonly string[]) {
  const [activeId, setActiveId] = useState("");

  useEffect(() => {
    if (ids.length === 0) return;

    const onScroll = () => {
      const atBottom =
        window.scrollY + window.innerHeight >=
        document.documentElement.scrollHeight - 1;
      setActiveId(
        resolveActiveHeading(
          readHeadingPositions(ids),
          window.innerHeight,
          atBottom,
        ),
      );
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [ids]);

  return activeId;
}

export const TableOfContents = ({
  entries,
}: {
  entries: readonly TocEntry[];
}) => {
  const headingIds = useMemo(
    () => entries.map(({ id }) => id),
    [entries],
  );
  const activeId = useActiveHeading(headingIds);

  if (entries.length === 0) return null;
  return (
    <nav className={s.nav} aria-label="Table of contents">
      <ul className={s.list}>
        {entries.map(({ id, text, depth }) => (
          <li key={id}>
            <a
              href={`#${id}`}
              onClick={(e) => {
                e.preventDefault();
                const el = document.getElementById(id);
                if (el) {
                  const reduceMotion = window.matchMedia(
                    "(prefers-reduced-motion: reduce)",
                  ).matches;
                  el.scrollIntoView({
                    behavior: reduceMotion ? "auto" : "smooth",
                  });
                  history.replaceState(null, "", `#${id}`);
                }
              }}
              aria-current={activeId === id ? "location" : undefined}
              className={classNames(
                s.link,
                depthClass(depth),
                activeId === id && s.linkActive,
              )}
            >
              {text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
};
