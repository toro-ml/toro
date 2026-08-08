import { style } from "@vanilla-extract/css";
import {
  color,
  fontSize,
  fontWeight,
  lineHeight,
  space,
} from "~/ui/tokens";

export const heroSection = style({
  display: "flex",
  flexDirection: "column",
  gap: space[1],
  marginBottom: space[8],
});

export const heroTitle = style({
  color: color.textEmphasis,
  fontSize: fontSize.display,
  fontWeight: fontWeight.bold,
  lineHeight: lineHeight.tight,
  margin: 0,
});

export const heroSubtitle = style({
  color: color.textMuted,
  fontSize: fontSize.md,
});
