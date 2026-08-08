import { style, styleVariants } from "@vanilla-extract/css";
import {
  color,
  duration,
  fontSize,
  fontWeight,
  layout,
  letterSpacing,
  radius,
  space,
} from "~/ui/tokens";

export const list = style({
  listStyle: "none",
  padding: 0,
  margin: 0,
  display: "flex",
  flexDirection: "column",
  gap: space[1],
});

const linkBase = style({
  display: "block",
  borderRadius: radius.sm,
  textDecoration: "none",
  color: color.text,
  transition: `color ${duration.normal}, background-color ${duration.normal}`,
  ":hover": {
    color: color.textEmphasis,
  },
});

export const link = styleVariants({
  sidebar: [
    linkBase,
    {
      padding: `${space[2]} ${layout.navPaddingInline}`,
      fontSize: fontSize.sm,
    },
  ],
  drawer: [
    linkBase,
    {
      padding: `${space[4]} ${space[5]}`,
      fontSize: fontSize.base,
    },
  ],
});

export const linkActive = style({
  backgroundColor: color.bgActive,
  color: color.textEmphasis,
  fontWeight: fontWeight.medium,
});

export const linkIndent = styleVariants({
  none: {},
  sidebarGroup: {
    paddingLeft: layout.navSubgroupIndent,
  },
  sidebarSubgroup: {
    paddingLeft: layout.navLinkIndent,
  },
  drawerGroup: {},
  drawerSubgroup: {},
});

const sectionHeadingBase = style({
  fontSize: fontSize.xs,
  fontWeight: fontWeight.semibold,
  textTransform: "uppercase",
  letterSpacing: letterSpacing.caps,
  color: color.textMuted,
});

export const sectionHeading = styleVariants({
  sidebar: [
    sectionHeadingBase,
    {
      padding: `${space[7]} ${layout.navPaddingInline} ${space[3]}`,
    },
  ],
  drawer: [
    sectionHeadingBase,
    {
      padding: `${space[7]} ${space[5]} ${space[3]}`,
    },
  ],
});

export const group = styleVariants({
  sidebar: {
    marginTop: space[2],
  },
  drawer: {},
});

const summaryBase = style({
  display: "flex",
  alignItems: "center",
  gap: space[3],
  color: color.textMuted,
  cursor: "pointer",
  borderRadius: radius.sm,
  listStyle: "none",
  transition: `color ${duration.normal}`,
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

export const groupSummary = styleVariants({
  sidebar: [
    summaryBase,
    {
      padding: `${space[2]} ${layout.navPaddingInline}`,
      fontSize: fontSize.sm,
      fontWeight: fontWeight.medium,
    },
  ],
  drawer: [
    summaryBase,
    {
      padding: `${space[3]} ${space[5]}`,
      fontSize: fontSize.base,
      fontWeight: fontWeight.medium,
    },
  ],
});

export const subgroupSummary = styleVariants({
  sidebar: [
    summaryBase,
    {
      padding: `${space[2]} ${layout.navPaddingInline} ${space[2]} ${layout.navSubgroupIndent}`,
      fontSize: fontSize.sm,
    },
  ],
  drawer: [
    summaryBase,
    {
      padding: `${space[3]} ${space[5]}`,
      fontSize: fontSize.base,
      fontWeight: fontWeight.medium,
    },
  ],
});

const chevronBase = style({
  flexShrink: 0,
  marginLeft: "auto",
  transition: `transform ${duration.fast} ease-out`,
  selectors: {
    "details[open] > summary > &": {
      transform: "rotate(90deg)",
    },
  },
});

export const chevron = styleVariants({
  sidebar: [
    chevronBase,
    {
      width: fontSize.sm,
      height: fontSize.sm,
    },
  ],
  drawer: [
    chevronBase,
    {
      width: "0.875rem",
      height: "0.875rem",
    },
  ],
});

export const groupItems = styleVariants({
  sidebar: {},
  drawer: {
    paddingLeft: layout.navIndent,
  },
});
