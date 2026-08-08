import { style, styleVariants } from "@vanilla-extract/css";
import { breakpoint, color } from "~/ui/tokens";

const baseStyle = style({
  position: "fixed",
  height: "var(--header-height)",
  display: "flex",
  alignItems: "center",
  backdropFilter: "blur(0.25rem)",
  top: 0,
  left: 0,
  right: 0,
  zIndex: 1000,
  overscrollBehaviorY: "contain",
  transition: "visibility 0.2s, opacity 0.2s",
});

export const headerStyle = styleVariants({
  hidden: [
    baseStyle,
    {
      visibility: "hidden",
      opacity: 0,
    },
  ],
  visible: [
    baseStyle,
    {
      visibility: "visible",
      opacity: 1,
    },
  ],
});

export const headerInner = style({
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  flex: 1,
  padding: "0 1rem",
});

export const headerRight = style({
  display: "flex",
  alignItems: "center",
  gap: "0.75rem",
});

export const menuButton = style({
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: "none",
  border: "none",
  color: "inherit",
  cursor: "pointer",
  padding: "0.25rem",
  "@media": {
    [breakpoint.sidebar]: {
      display: "none",
    },
  },
});

export const logoLink = style({
  display: "flex",
  alignItems: "center",
  gap: "0.75rem",
  textDecoration: "none",
  color: "inherit",
});

export const logoImage = style({
  borderRadius: "50%",
});

export const iconLink = style({
  display: "flex",
  color: "inherit",
});

const drawerBase = style({
  position: "fixed",
  top: 0,
  left: 0,
  right: 0,
  backdropFilter: "blur(0.25rem)",
  paddingTop: "var(--header-height)",
  paddingLeft: "1.5rem",
  paddingRight: "1.5rem",
  paddingBottom: "1rem",
  overflowY: "auto",
  zIndex: 999,
  transition: "transform 0.25s ease, opacity 0.25s ease",
  transformOrigin: "top",
});

export const drawer = styleVariants({
  open: [drawerBase, { transform: "translateY(0)", opacity: 1 }],
  closed: [drawerBase, { transform: "translateY(-100%)", opacity: 0 }],
});

const backdropBase = style({
  position: "fixed",
  inset: 0,
  zIndex: 998,
  backgroundColor: "rgba(0, 0, 0, 0.5)",
  transition: "opacity 0.2s ease",
});

export const backdrop = styleVariants({
  open: [backdropBase, { opacity: 1 }],
  closed: [backdropBase, { opacity: 0, pointerEvents: "none" }],
});
