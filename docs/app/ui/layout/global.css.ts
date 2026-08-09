import { grayDark, whiteA } from "@radix-ui/colors";
import { globalStyle, keyframes } from "@vanilla-extract/css";
import {
  color,
  duration,
  easing,
  fontFamily,
  layout,
  lineHeight,
  space,
} from "~/ui/tokens";

globalStyle(":root", {
  vars: {
    "--header-height": layout.headerHeight,
    "--header-offset": `calc(${layout.headerHeight} + ${space[4]})`,
    "--color-text": whiteA.whiteA9,
    "--color-text-emphasis": whiteA.whiteA12,
    "--color-text-muted": whiteA.whiteA8,
    "--color-bg": grayDark.gray2,
    "--color-bg-code": grayDark.gray3,
    "--color-bg-active": grayDark.gray4,
    "--color-backdrop": "rgba(0, 0, 0, 0.5)",
    "--color-border": grayDark.gray4,
    "--color-border-strong": grayDark.gray5,
    "--color-border-emphasis": grayDark.gray6,
    "--color-focus": "#93c5fd",
    "--color-link": "#93c5fd",
  },
  scrollBehavior: "smooth",
  scrollPaddingTop: "var(--header-offset)",
  "@media": {
    "(prefers-reduced-motion: reduce)": {
      scrollBehavior: "auto",
    },
  },
});

globalStyle(`:root[data-direction]`, {
  scrollBehavior: "auto",
});

globalStyle("html", {
  backgroundColor: grayDark.gray2,
  colorScheme: "dark",
});

globalStyle("body", {
  margin: 0,
});

globalStyle(".radix-themes", {
  backgroundColor: "var(--color-bg)",
  color: "var(--color-text)",
  lineHeight: lineHeight.normal,
  fontFamily: fontFamily.sans,
  minHeight: "100vh",
});

globalStyle(":where(a, button, summary):focus-visible", {
  outline: `2px solid ${color.focus}`,
  outlineOffset: space[1],
});

const slideOutToLeft = keyframes({
  to: { opacity: 0, transform: "translateX(-2rem)" },
});
const slideInFromRight = keyframes({
  from: { opacity: 0, transform: "translateX(2rem)" },
});
const slideOutToRight = keyframes({
  to: { opacity: 0, transform: "translateX(2rem)" },
});
const slideInFromLeft = keyframes({
  from: { opacity: 0, transform: "translateX(-2rem)" },
});

globalStyle("::view-transition-old(doc-content)", {
  animationDuration: duration.normal,
  animationTimingFunction: easing.exit,
  "@media": {
    "(prefers-reduced-motion: reduce)": {
      animationDuration: "0.01ms",
    },
  },
});

globalStyle("::view-transition-new(doc-content)", {
  animationDuration: duration.slow,
  animationTimingFunction: easing.enter,
  "@media": {
    "(prefers-reduced-motion: reduce)": {
      animationDuration: "0.01ms",
    },
  },
});

globalStyle(
  `:root[data-direction="forward"]::view-transition-old(doc-content)`,
  { animationName: slideOutToLeft },
);
globalStyle(
  `:root[data-direction="forward"]::view-transition-new(doc-content)`,
  { animationName: slideInFromRight },
);
globalStyle(
  `:root[data-direction="back"]::view-transition-old(doc-content)`,
  { animationName: slideOutToRight },
);
globalStyle(
  `:root[data-direction="back"]::view-transition-new(doc-content)`,
  { animationName: slideInFromLeft },
);
