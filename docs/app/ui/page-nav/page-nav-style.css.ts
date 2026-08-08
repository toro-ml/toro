import { style } from "@vanilla-extract/css";
import {
  borderWidth,
  color,
  duration,
  fontSize,
  fontWeight,
  letterSpacing,
  space,
} from "~/ui/tokens";

export const root = style({
  display: "flex",
  justifyContent: "space-between",
  gap: space[6],
  borderTop: `${borderWidth.default} solid ${color.border}`,
  marginTop: space[8],
  paddingTop: space[7],
});

export const link = style({
  display: "flex",
  flexDirection: "column",
  gap: space[1],
  textDecoration: "none",
  color: color.text,
  transition: `color ${duration.normal}`,
  ":hover": {
    color: color.textEmphasis,
  },
});

export const next = style([
  link,
  { alignItems: "flex-end", marginLeft: "auto" },
]);

export const label = style({
  fontSize: fontSize.xs,
  color: color.textMuted,
  textTransform: "uppercase",
  letterSpacing: letterSpacing.caps,
});

export const title = style({
  fontSize: fontSize.base,
  fontWeight: fontWeight.medium,
});
