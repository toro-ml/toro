import { style } from "@vanilla-extract/css";
import { breakpoint, color } from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: "var(--header-height)",
  height: "calc(100dvh - var(--header-height))",
  padding: "1.5rem 1rem 1.5rem 0.75rem",
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
  fontSize: "0.6875rem",
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  color: color.textEmphasis,
  marginBottom: "0.5rem",
  padding: "0 0.75rem",
});

export const list = style({
  listStyle: "none",
  padding: 0,
  margin: 0,
});

export const link = style({
  display: "block",
  position: "relative",
  padding: "0.375rem 0.75rem",
  fontSize: "0.8125rem",
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
  paddingLeft: "1.5rem",
});
