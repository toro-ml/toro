import { style } from "@vanilla-extract/css";
import { borderWidth, breakpoint, color, space } from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: 0,
  height: "100dvh",
  boxSizing: "border-box",
  padding: `calc(var(--header-height) + ${space[6]}) ${space[4]} ${space[9]} ${space[5]}`,
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
