import { style } from "@vanilla-extract/css";
import { breakpoint, color } from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: "var(--header-height)",
  height: "calc(100dvh - var(--header-height))",
  padding: "1.5rem 0.75rem 1.5rem 1rem",
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
  gap: "0.125rem",
});

export const link = style({
  display: "block",
  padding: "0.375rem 0.75rem",
  borderRadius: "0.25rem",
  textDecoration: "none",
  fontSize: "0.875rem",
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
