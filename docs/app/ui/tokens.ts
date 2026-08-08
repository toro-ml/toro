export const color = {
  text: "var(--color-text)",
  textEmphasis: "var(--color-text-emphasis)",
  textMuted: "var(--color-text-muted)",
  bg: "var(--color-bg)",
  bgCode: "var(--color-bg-code)",
  bgActive: "var(--color-bg-active)",
  backdrop: "var(--color-backdrop)",
  border: "var(--color-border)",
  borderStrong: "var(--color-border-strong)",
  borderEmphasis: "var(--color-border-emphasis)",
  focus: "var(--color-focus)",
  link: "var(--color-link)",
} as const;

export const fontFamily = {
  sans: '"Noto Sans JP", sans-serif',
  mono: '"Fira Code", ui-monospace, monospace',
} as const;

export const fontSize = {
  xs: "0.6875rem",
  sm: "0.8125rem",
  base: "0.9375rem",
  md: "1rem",
  code: "0.8125rem",
  lg: "1.125rem",
  xl: "1.375rem",
  "2xl": "1.75rem",
  display: "3.75rem",
} as const;

export const fontWeight = {
  regular: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
} as const;

export const lineHeight = {
  tight: "1.3",
  normal: "1.6",
  code: "1.5",
} as const;

export const letterSpacing = {
  caps: "0.05em",
} as const;

export const space = {
  1: "0.125rem",
  2: "0.25rem",
  3: "0.375rem",
  4: "0.5rem",
  5: "0.75rem",
  6: "1rem",
  7: "1.5rem",
  8: "2rem",
  9: "3rem",
} as const;

export const radius = {
  sm: "0.25rem",
  md: "0.375rem",
  full: "9999px",
} as const;

export const borderWidth = {
  default: "1px",
  strong: "2px",
  emphasis: "3px",
} as const;

export const duration = {
  fast: "100ms",
  normal: "150ms",
  slow: "250ms",
} as const;

export const easing = {
  standard: "ease",
  enter: "ease-in",
  exit: "ease-out",
} as const;

export const zIndex = {
  backdrop: 998,
  drawer: 999,
  header: 1000,
} as const;

export const layout = {
  headerHeight: "3rem",
  sidebarWidth: "14rem",
  tocWidth: "14rem",
  navPaddingInline: "0.625rem",
  navIndent: "1.25rem",
  navSubgroupIndent: "1.75rem",
  navLinkIndent: "2.875rem",
  tocDepth3Indent: "2rem",
  tocDepth4Indent: "2.75rem",
} as const;

export const breakpoint = {
  sidebar: "(min-width: 50rem)",
  toc: "(min-width: 72rem)",
} as const;
