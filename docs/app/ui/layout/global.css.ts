import { grayDark, whiteA } from "@radix-ui/colors";
import { globalStyle } from "@vanilla-extract/css";

globalStyle(":root", {
  vars: {
    "--header-height": "3.5rem",
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
});

globalStyle("body", {
  margin: 0,
});

globalStyle(".radix-themes", {
  backgroundColor: "var(--color-bg)",
  color: "var(--color-text)",
  lineHeight: 1.7,
  fontFamily: '"Noto Sans JP", sans-serif',
  minHeight: "100vh",
});
