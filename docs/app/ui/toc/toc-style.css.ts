import { style } from "@vanilla-extract/css";
import { breakpoint, color, fontSize, space } from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: "var(--header-height)",
  height: "calc(100dvh - var(--header-height))",
  padding: `${space[6]} ${space[5]} ${space[6]} ${space[4]}`,
  overflowY: "auto",
  borderLeft: `1px solid ${color.border}`,
  display: "none",
  "@media": {
    [breakpoint.toc]: {
      display: "block",
    },
  },
});

export const heading = style({
  fontSize: fontSize.xs,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  color: color.textEmphasis,
  marginBottom: space[3],
  padding: `0 0.625rem`,
});

export const list = style({
  listStyle: "none",
  padding: 0,
  margin: 0,
});

export const link = style({
  display: "block",
  position: "relative",
  padding: `${space[2]} 0.625rem`,
  fontSize: fontSize.sm,
  lineHeight: 1.5,
  color: color.textMuted,
  textDecoration: "none",
  transition: "color 0.15s",
  "::before": {
    content: '""',
    position: "absolute",
    left: 0,
    top: "10%",
    height: "80%",
    width: "2px",
    backgroundColor: "transparent",
    transition: "background-color 0.15s",
  },
  ":hover": {
    color: color.textEmphasis,
  },
});

export const linkActive = style({
  color: color.textEmphasis,
  "::before": {
    backgroundColor: color.textEmphasis,
  },
});

export const linkDepth3 = style({
  paddingLeft: space[7],
});
