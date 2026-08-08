import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";
import { Links, Meta, Outlet, Scripts, ScrollRestoration } from "react-router";
import { Header } from "~/ui/header";
import "~/ui/layout/global.css";

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      style={{ scrollPaddingTop: "3.5rem" }}
    >
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/toro/img/favicon.png" />
        <Meta />
        <Links />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500;700&family=Fira+Code:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Theme appearance="dark">
          <Header />
          {children}
        </Theme>
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

export default function App() {
  return <Outlet />;
}
