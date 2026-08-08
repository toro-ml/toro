import { style } from "@vanilla-extract/css";
import { breakpoint, color, fontSize, space } from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: "var(--header-height)",
  height: "calc(100dvh - var(--header-height))",
  padding: `${space[6]} ${space[4]} ${space[6]} ${space[5]}`,
  overflowY: "auto",
  borderRight: `1px solid ${color.border}`,
  display: "none",
  "@media": {
    [breakpoint.sidebar]: {
      display: "block",
    },
  },
});

export const list = style({
  listStyle: "none",
  padding: 0,
  margin: 0,
  display: "flex",
  flexDirection: "column",
  gap: space[1],
});

export const link = style({
  display: "block",
  padding: `${space[2]} 0.625rem`,
  borderRadius: space[2],
  textDecoration: "none",
  fontSize: fontSize.sm,
  color: color.text,
  transition: "color 0.15s, background-color 0.15s",
  ":hover": {
    color: color.textEmphasis,
  },
});

export const linkActive = style([
  link,
  {
    backgroundColor: color.bgActive,
    color: color.textEmphasis,
    fontWeight: 500,
  },
]);

export const sectionHeading = style({
  fontSize: fontSize.xs,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  color: color.textMuted,
  padding: `${space[7]} 0.625rem ${space[3]}`,
});
