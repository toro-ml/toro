import { globalStyle, style } from "@vanilla-extract/css";
import { breakpoint, color, fontSize, space } from "~/ui/tokens";

export const nav = style({
  position: "sticky",
  top: "var(--header-height)",
  height: "calc(100dvh - var(--header-height))",
  boxSizing: "border-box",
  padding: `${space[6]} ${space[4]} ${space[9]} ${space[5]}`,
  overflowY: "auto",
  overscrollBehavior: "contain",
  borderRight: `1px solid ${color.border}`,
  display: "none",
  interpolateSize: "allow-keywords",
  "@media": {
    [breakpoint.sidebar]: {
      display: "block",
    },
  },
} as any);

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

export const group = style({
  marginTop: space[2],
});

export const groupSummary = style({
  display: "flex",
  alignItems: "center",
  gap: space[3],
  padding: `${space[2]} 0.625rem`,
  fontSize: fontSize.sm,
  fontWeight: 500,
  color: color.textMuted,
  cursor: "pointer",
  borderRadius: space[2],
  transition: "color 0.15s",
  listStyle: "none",
  ":hover": {
    color: color.textEmphasis,
  },
  selectors: {
    "&::marker": { display: "none" },
    "&::-webkit-details-marker": { display: "none" },
    "details[open] > &": {
      color: color.text,
    },
  },
});

export const groupChevron = style({
  width: "0.75rem",
  height: "0.75rem",
  flexShrink: 0,
  marginLeft: "auto",
  transition: "transform 0.10s ease-out",
  selectors: {
    "details[open] > summary > &": {
      transform: "rotate(90deg)",
    },
  },
});

export const groupItems = style({});

export const subGroup = style({
  marginTop: space[1],
});

export const subGroupSummary = style({
  display: "flex",
  alignItems: "center",
  gap: space[3],
  padding: `${space[2]} 0.625rem ${space[2]} 1.75rem`,
  fontSize: fontSize.sm,
  color: color.textMuted,
  cursor: "pointer",
  borderRadius: space[2],
  transition: "color 0.15s",
  listStyle: "none",
  ":hover": {
    color: color.textEmphasis,
  },
  selectors: {
    "&::marker": { display: "none" },
    "&::-webkit-details-marker": { display: "none" },
    "details[open] > &": {
      color: color.text,
    },
  },
});

export const subGroupItems = style({});

globalStyle(`${groupItems} > div > .${list} > li > .${link}`, {
  paddingLeft: "1.75rem",
});

globalStyle(`${subGroupItems} > .${list} > li > .${link}`, {
  paddingLeft: "2.875rem",
});

globalStyle(`${group}::details-content, ${subGroup}::details-content`, {
  height: 0,
  overflow: "clip",
  transition: "height 0.15s ease-out, content-visibility 0.10s allow-discrete",
} as any);

globalStyle(
  `${group}[open]::details-content, ${subGroup}[open]::details-content`,
  { height: "auto" },
);
