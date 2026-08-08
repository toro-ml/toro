export const color = {
  text: "var(--color-text)",
  textEmphasis: "var(--color-text-emphasis)",
  textMuted: "var(--color-text-muted)",
  bg: "var(--color-bg)",
  bgCode: "var(--color-bg-code)",
  bgActive: "var(--color-bg-active)",
  border: "var(--color-border)",
  borderStrong: "var(--color-border-strong)",
  borderEmphasis: "var(--color-border-emphasis)",
  link: "var(--color-link)",
} as const;

export const fontSize = {
  xs: "0.6875rem",
  sm: "0.8125rem",
  base: "0.9375rem",
  code: "0.8125rem",
  lg: "1.125rem",
  xl: "1.375rem",
  "2xl": "1.75rem",
} as const;

export const lineHeight = {
  tight: "1.3",
  normal: "1.6",
  code: "1.5",
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

export const breakpoint = {
  sidebar: "(min-width: 50rem)",
  toc: "(min-width: 72rem)",
} as const;
