import { style } from "@vanilla-extract/css";
import {
  borderWidth,
  breakpoint,
  color,
  duration,
  fontSize,
  layout,
  lineHeight,
  space,
} from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: 0,
  height: "100dvh",
  boxSizing: "border-box",
  marginTop: "calc(0px - var(--header-height))",
  padding: `calc(var(--header-height) + ${space[6]}) ${space[5]} ${space[6]} ${space[4]}`,
  overflowY: "auto",
  overscrollBehavior: "contain",
  borderLeft: `${borderWidth.default} solid ${color.border}`,
  display: "none",
  "@media": {
    [breakpoint.toc]: {
      display: "block",
    },
  },
});

export const list = style({
  listStyle: "none",
  padding: 0,
  margin: 0,
});

export const link = style({
  display: "block",
  position: "relative",
  padding: `${space[2]} ${layout.navPaddingInline}`,
  fontSize: fontSize.sm,
  lineHeight: lineHeight.code,
  color: color.textMuted,
  textDecoration: "none",
  transition: `color ${duration.normal}`,
  "::before": {
    content: '""',
    position: "absolute",
    left: 0,
    top: "10%",
    height: "80%",
    width: borderWidth.strong,
    backgroundColor: "transparent",
    transition: `background-color ${duration.normal}`,
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

export const linkDepth2 = style({
  paddingLeft: layout.navIndent,
});

export const linkDepth3 = style({
  paddingLeft: layout.tocDepth3Indent,
});

export const linkDepth4 = style({
  paddingLeft: layout.tocDepth4Indent,
});
