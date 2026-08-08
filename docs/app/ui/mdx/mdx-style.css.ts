import { globalStyle, style } from "@vanilla-extract/css";
import { color, fontSize, lineHeight, space } from "~/ui/tokens";

export const article = style({
  minWidth: 0,
  padding: `1.25rem ${space[7]} ${space[9]}`,
  fontSize: fontSize.base,
  viewTransitionName: "doc-content",
});

globalStyle(`${article} :is(h1, h2, h3)`, {
  scrollMarginTop: "var(--header-height)",
});

globalStyle(`${article} h1`, {
  fontSize: fontSize["2xl"],
  fontWeight: 700,
  marginTop: 0,
  marginBottom: space[5],
  color: color.textEmphasis,
});

globalStyle(`${article} h2`, {
  fontSize: fontSize.xl,
  fontWeight: 600,
  marginTop: space[8],
  marginBottom: space[4],
  color: color.textEmphasis,
  borderBottom: `1px solid ${color.border}`,
  paddingBottom: space[2],
});

globalStyle(`${article} h3`, {
  fontSize: fontSize.lg,
  fontWeight: 600,
  marginTop: space[7],
  marginBottom: space[3],
  color: color.textEmphasis,
});

globalStyle(`${article} p`, {
  marginTop: 0,
  marginBottom: space[5],
});

globalStyle(`${article} a`, {
  color: color.link,
  textDecoration: "none",
});

globalStyle(`${article} a:hover`, {
  textDecoration: "underline",
});

globalStyle(`${article} code`, {
  fontFamily: '"Fira Code", ui-monospace, monospace',
  fontSize: "0.875em",
  backgroundColor: color.bgCode,
  padding: `${space[1]} ${space[3]}`,
  borderRadius: space[2],
});

globalStyle(`${article} pre`, {
  fontFamily: '"Fira Code", ui-monospace, monospace',
  fontSize: fontSize.code,
  backgroundColor: color.bgCode,
  padding: `${space[5]} ${space[6]}`,
  borderRadius: space[3],
  overflowX: "auto",
  lineHeight: lineHeight.code,
  margin: `0 0 ${space[6]}`,
});

globalStyle(`${article} pre code`, {
  backgroundColor: "transparent",
  padding: 0,
  borderRadius: 0,
  fontSize: "inherit",
});

globalStyle(`${article} ul, ${article} ol`, {
  paddingLeft: "1.25rem",
  marginBottom: space[5],
});

globalStyle(`${article} li`, {
  marginBottom: space[1],
});

globalStyle(`${article} table`, {
  width: "100%",
  borderCollapse: "collapse",
  marginBottom: space[6],
  fontSize: fontSize.sm,
});

globalStyle(`${article} th`, {
  textAlign: "left",
  borderBottom: `2px solid ${color.borderStrong}`,
  padding: `${space[3]} 0.625rem`,
  fontWeight: 600,
  color: color.textEmphasis,
});

globalStyle(`${article} td`, {
  borderBottom: `1px solid ${color.border}`,
  padding: `${space[3]} 0.625rem`,
});

globalStyle(`${article} blockquote`, {
  borderLeft: `3px solid ${color.borderEmphasis}`,
  margin: `0 0 ${space[6]}`,
  padding: `${space[3]} ${space[5]}`,
  color: color.text,
});

globalStyle(`${article} hr`, {
  border: "none",
  borderTop: `1px solid ${color.border}`,
  margin: `${space[7]} 0`,
});

export const pageNav = style({
  display: "flex",
  justifyContent: "space-between",
  gap: space[6],
  borderTop: `1px solid ${color.border}`,
  marginTop: space[8],
  paddingTop: space[7],
});

export const pageNavLink = style({
  display: "flex",
  flexDirection: "column",
  gap: space[1],
  textDecoration: "none",
  color: color.text,
  transition: "color 0.15s",
  ":hover": {
    color: color.textEmphasis,
  },
});

export const pageNavNext = style([
  pageNavLink,
  { alignItems: "flex-end", marginLeft: "auto" },
]);

export const pageNavLabel = style({
  fontSize: fontSize.xs,
  color: color.textMuted,
  textTransform: "uppercase",
  letterSpacing: "0.05em",
});

export const pageNavTitle = style({
  fontSize: fontSize.base,
  fontWeight: 500,
});
