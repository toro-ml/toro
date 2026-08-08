import { useMemo, type AnchorHTMLAttributes } from "react";
import { Link } from "react-router";
import * as runtime from "react/jsx-runtime";

function MdxLink(props: AnchorHTMLAttributes<HTMLAnchorElement>) {
  const { href, ...rest } = props;
  if (href && href.startsWith("/") && !href.startsWith("//")) {
    return <Link to={href} {...rest} />;
  }
  return <a href={href} {...rest} />;
}

const components = { a: MdxLink };

export function MdxContent({ code }: { code: string }) {
  const Component = useMemo(() => {
    const fn = new Function(code);
    return fn({
      ...runtime,
      baseUrl: import.meta.url,
    }).default;
  }, [code]);

  return <Component components={components} />;
}
