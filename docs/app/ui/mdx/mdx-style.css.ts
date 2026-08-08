import { globalStyle, style } from "@vanilla-extract/css";
import { color } from "~/ui/tokens";

export const article = style({
  minWidth: 0,
  padding: "1.5rem 2rem 4rem",
  fontSize: "1.0625rem",
});

globalStyle(`${article} :is(h1, h2, h3)`, {
  scrollMarginTop: "var(--header-height)",
});

globalStyle(`${article} h1`, {
  fontSize: "2rem",
  fontWeight: 700,
  marginTop: 0,
  marginBottom: "1rem",
  color: color.textEmphasis,
});

globalStyle(`${article} h2`, {
  fontSize: "1.5rem",
  fontWeight: 600,
  marginTop: "2.5rem",
  marginBottom: "0.75rem",
  color: color.textEmphasis,
  borderBottom: `1px solid ${color.border}`,
  paddingBottom: "0.375rem",
});

globalStyle(`${article} h3`, {
  fontSize: "1.25rem",
  fontWeight: 600,
  marginTop: "2rem",
  marginBottom: "0.5rem",
  color: color.textEmphasis,
});

globalStyle(`${article} p`, {
  marginTop: 0,
  marginBottom: "1rem",
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
  padding: "0.125rem 0.375rem",
  borderRadius: "0.25rem",
});

globalStyle(`${article} pre`, {
  fontFamily: '"Fira Code", ui-monospace, monospace',
  fontSize: "0.875rem",
  backgroundColor: color.bgCode,
  padding: "1rem 1.25rem",
  borderRadius: "0.5rem",
  overflowX: "auto",
  lineHeight: 1.6,
  margin: "0 0 1rem",
});

globalStyle(`${article} pre code`, {
  backgroundColor: "transparent",
  padding: 0,
  borderRadius: 0,
  fontSize: "inherit",
});

globalStyle(`${article} ul, ${article} ol`, {
  paddingLeft: "1.5rem",
  marginBottom: "1rem",
});

globalStyle(`${article} li`, {
  marginBottom: "0.25rem",
});

globalStyle(`${article} table`, {
  width: "100%",
  borderCollapse: "collapse",
  marginBottom: "1rem",
  fontSize: "0.9375rem",
});

globalStyle(`${article} th`, {
  textAlign: "left",
  borderBottom: `2px solid ${color.borderStrong}`,
  padding: "0.5rem 0.75rem",
  fontWeight: 600,
  color: color.textEmphasis,
});

globalStyle(`${article} td`, {
  borderBottom: `1px solid ${color.border}`,
  padding: "0.5rem 0.75rem",
});

globalStyle(`${article} blockquote`, {
  borderLeft: `3px solid ${color.borderEmphasis}`,
  margin: "0 0 1rem",
  padding: "0.5rem 1rem",
  color: color.text,
});

globalStyle(`${article} hr`, {
  border: "none",
  borderTop: `1px solid ${color.border}`,
  margin: "2rem 0",
});
