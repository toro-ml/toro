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

export const breakpoint = {
  sidebar: "(min-width: 50rem)",
  toc: "(min-width: 72rem)",
} as const;
