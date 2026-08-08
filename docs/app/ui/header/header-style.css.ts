import { style, styleVariants } from "@vanilla-extract/css";

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
