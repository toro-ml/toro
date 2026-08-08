import { style } from "@vanilla-extract/css";
import { breakpoint, layout } from "~/ui/tokens";

export const docsGrid = style({
  display: "grid",
  gridTemplateColumns: "1fr",
  minHeight: "100vh",
  paddingTop: "var(--header-height)",
  "@media": {
    [breakpoint.sidebar]: {
      gridTemplateColumns: `${layout.sidebarWidth} minmax(0, 1fr)`,
    },
  },
});

export const docsMain = style({
  display: "grid",
  gridTemplateColumns: "minmax(0, 1fr)",
  minWidth: 0,
  alignItems: "start",
  "@media": {
    [breakpoint.toc]: {
      gridTemplateColumns: `minmax(0, 1fr) ${layout.tocWidth}`,
    },
  },
});
