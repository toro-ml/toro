import { style } from "@vanilla-extract/css";
import { breakpoint } from "~/ui/tokens";

export const docsGrid = style({
  display: "grid",
  gridTemplateColumns: "1fr",
  minHeight: "100vh",
  paddingTop: "var(--header-height)",
  "@media": {
    [breakpoint.sidebar]: {
      gridTemplateColumns: "15rem 1fr",
    },
    [breakpoint.toc]: {
      gridTemplateColumns: "15rem 1fr 15rem",
    },
  },
});
