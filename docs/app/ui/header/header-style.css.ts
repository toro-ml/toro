import { style, styleVariants } from "@vanilla-extract/css";
import {
  breakpoint,
  color,
  duration,
  fontSize,
  fontWeight,
  layout,
  radius,
  space,
  zIndex,
} from "~/ui/tokens";

export const headerStyle = style({
  position: "fixed",
  height: "var(--header-height)",
  display: "flex",
  alignItems: "center",
  backdropFilter: `blur(${space[2]})`,
  top: 0,
  left: 0,
  right: 0,
  zIndex: zIndex.header,
});

export const headerInner = style({
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  flex: 1,
  padding: `0 ${space[5]}`,
  "@media": {
    [breakpoint.sidebar]: {
      paddingInline: fontSize.xl,
    },
  },
});

export const headerRight = style({
  display: "flex",
  alignItems: "center",
  gap: space[5],
});

const iconControl = style({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minWidth: space[8],
  minHeight: space[8],
  borderRadius: radius.sm,
  color: "inherit",
});

export const menuButton = style([
  iconControl,
  {
    background: "none",
    border: "none",
    cursor: "pointer",
    padding: space[2],
    "@media": {
      [breakpoint.sidebar]: {
        display: "none",
      },
    },
  },
]);

export const logoLink = style({
  display: "flex",
  alignItems: "center",
  gap: space[5],
  textDecoration: "none",
  color: "inherit",
});

export const logoImage = style({
  borderRadius: radius.full,
});

export const logoTitle = style({
  fontSize: fontSize.lg,
  fontWeight: fontWeight.bold,
  color: color.textEmphasis,
});

export const iconLink = style([
  iconControl,
  {
    textDecoration: "none",
  },
]);

const drawerBase = style({
  position: "fixed",
  inset: 0,
  backdropFilter: `blur(${space[2]})`,
  paddingTop: "var(--header-height)",
  paddingInline: space[7],
  paddingBottom: space[6],
  overflowY: "auto",
  zIndex: zIndex.drawer,
  transition: `transform ${duration.slow} ease, opacity ${duration.slow} ease`,
  transformOrigin: "top",
  "@media": {
    [breakpoint.sidebar]: {
      display: "none",
    },
    "(prefers-reduced-motion: reduce)": {
      transitionDuration: "0.01ms",
    },
  },
});

export const drawer = styleVariants({
  open: [drawerBase, { transform: "translateY(0)", opacity: 1 }],
  closed: [drawerBase, { transform: "translateY(-100%)", opacity: 0 }],
});

const backdropBase = style({
  position: "fixed",
  inset: 0,
  zIndex: zIndex.backdrop,
  backgroundColor: color.backdrop,
  transition: `opacity ${duration.slow} ease`,
  "@media": {
    [breakpoint.sidebar]: {
      display: "none",
    },
    "(prefers-reduced-motion: reduce)": {
      transitionDuration: "0.01ms",
    },
  },
});

export const backdrop = styleVariants({
  open: [backdropBase, { opacity: 1 }],
  closed: [backdropBase, { opacity: 0, pointerEvents: "none" }],
});
