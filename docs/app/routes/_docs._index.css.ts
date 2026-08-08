import { style } from "@vanilla-extract/css";
import { color } from "~/ui/tokens";

export const heroSection = style({
  marginBottom: "2rem",
});

export const heroTitle = style({
  color: color.textEmphasis,
});

export const heroSubtitle = style({
  opacity: 0.7,
});
