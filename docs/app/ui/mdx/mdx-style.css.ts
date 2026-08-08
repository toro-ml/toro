import { globalStyle, style } from "@vanilla-extract/css";
import {
  borderWidth,
  color,
  fontFamily,
  fontSize,
  fontWeight,
  layout,
  lineHeight,
  radius,
  space,
} from "~/ui/tokens";

export const article = style({
  minWidth: 0,
  padding: `${layout.navIndent} ${space[7]} ${space[9]}`,
  fontSize: fontSize.base,
  viewTransitionName: "doc-content",
});

globalStyle(`${article} :is(h1, h2, h3)`, {
  scrollMarginTop: "var(--header-offset)",
});

globalStyle(`${article} h1`, {
  fontSize: fontSize["2xl"],
  fontWeight: fontWeight.bold,
  marginTop: 0,
  marginBottom: space[5],
  color: color.textEmphasis,
});

globalStyle(`${article} h2`, {
  fontSize: fontSize.xl,
  fontWeight: fontWeight.semibold,
  marginTop: space[8],
  marginBottom: space[4],
  color: color.textEmphasis,
  borderBottom: `${borderWidth.default} solid ${color.border}`,
  paddingBottom: space[2],
});

globalStyle(`${article} h3`, {
  fontSize: fontSize.lg,
  fontWeight: fontWeight.semibold,
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
  fontFamily: fontFamily.mono,
  fontSize: "0.875em",
  backgroundColor: color.bgCode,
  padding: `${space[1]} ${space[3]}`,
  borderRadius: radius.sm,
});

globalStyle(`${article} pre`, {
  fontFamily: fontFamily.mono,
  fontSize: fontSize.code,
  backgroundColor: color.bgCode,
  padding: `${space[5]} ${space[6]}`,
  borderRadius: radius.md,
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
  paddingLeft: layout.navIndent,
  marginBottom: space[5],
});

globalStyle(`${article} li`, {
  marginBottom: space[1],
});

globalStyle(`${article} table`, {
  width: "100%",
  tableLayout: "fixed",
  borderCollapse: "collapse",
  marginBottom: space[6],
  fontSize: fontSize.sm,
});

globalStyle(`${article} th`, {
  textAlign: "left",
  borderBottom: `${borderWidth.strong} solid ${color.borderStrong}`,
  padding: `${space[3]} ${layout.navPaddingInline}`,
  fontWeight: fontWeight.semibold,
  color: color.textEmphasis,
});

globalStyle(`${article} th:first-child, ${article} td:first-child`, {
  width: "45%",
});

globalStyle(`${article} td`, {
  borderBottom: `${borderWidth.default} solid ${color.border}`,
  padding: `${space[4]} ${layout.navPaddingInline}`,
  wordBreak: "break-word",
  lineHeight: lineHeight.normal,
});

globalStyle(`${article} td code`, {
  whiteSpace: "normal",
  wordBreak: "break-all",
});

globalStyle(`${article} td .katex`, {
  fontSize: "1.1em",
});

globalStyle(`${article} td .katex-html`, {
  overflow: "visible",
});

globalStyle(`${article} blockquote`, {
  borderLeft: `${borderWidth.emphasis} solid ${color.borderEmphasis}`,
  margin: `0 0 ${space[6]}`,
  padding: `${space[3]} ${space[5]}`,
  color: color.text,
});

globalStyle(`${article} hr`, {
  border: "none",
  borderTop: `${borderWidth.default} solid ${color.border}`,
  margin: `${space[7]} 0`,
});

export const namespaceBadge = style({
  fontSize: fontSize.sm,
  color: color.textMuted,
  marginBottom: space[2],
});
