import { grayDark, whiteA } from "@radix-ui/colors";
import { globalStyle, keyframes } from "@vanilla-extract/css";
import { lineHeight, space } from "~/ui/tokens";

globalStyle(":root", {
  vars: {
    "--header-height": space[9],
    "--color-text": whiteA.whiteA9,
    "--color-text-emphasis": whiteA.whiteA12,
    "--color-text-muted": whiteA.whiteA8,
    "--color-bg": grayDark.gray2,
    "--color-bg-code": grayDark.gray3,
    "--color-bg-active": grayDark.gray4,
    "--color-border": grayDark.gray4,
    "--color-border-strong": grayDark.gray5,
    "--color-border-emphasis": grayDark.gray6,
    "--color-link": "#93c5fd",
  },
  scrollBehavior: "smooth",
});

globalStyle(`:root[data-direction]`, {
  scrollBehavior: "auto",
});

globalStyle("body", {
  margin: 0,
});

globalStyle(".radix-themes", {
  backgroundColor: "var(--color-bg)",
  color: "var(--color-text)",
  lineHeight: lineHeight.normal,
  fontFamily: '"Noto Sans JP", sans-serif',
  minHeight: "100vh",
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
  animationDuration: "150ms",
  animationTimingFunction: "ease-out",
});

globalStyle("::view-transition-new(doc-content)", {
  animationDuration: "200ms",
  animationTimingFunction: "ease-in",
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
