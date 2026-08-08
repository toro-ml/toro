import { style } from "@vanilla-extract/css";
import { borderWidth, breakpoint, color, space } from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: "var(--header-height)",
  height: "calc(100dvh - var(--header-height))",
  boxSizing: "border-box",
  padding: `${space[6]} ${space[4]} ${space[9]} ${space[5]}`,
  overflowY: "auto",
  overscrollBehavior: "contain",
  borderRight: `${borderWidth.default} solid ${color.border}`,
  display: "none",
  "@media": {
    [breakpoint.sidebar]: {
      display: "block",
    },
  },
});
