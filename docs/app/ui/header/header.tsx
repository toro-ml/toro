import {
  Cross1Icon,
  GitHubLogoIcon,
  HamburgerMenuIcon,
} from "@radix-ui/react-icons";
import { Heading } from "@radix-ui/themes";
import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router";
import { navSections } from "~/ui/sidebar";
import type { NavGroupEntry } from "~/ui/sidebar";
import {
  backdrop,
  drawer,
  drawerGroupChevron,
  drawerGroupItems,
  drawerGroupSummary,
  drawerLink,
  drawerLinkActive,
  drawerSectionHeading,
  headerInner,
  headerRight,
  headerStyle,
  iconLink,
  logoImage,
  logoLink,
  menuButton,
} from "./header-style.css";

const ChevronIcon = ({ className }: { className?: string }) => (
  <svg
    viewBox="0 0 16 16"
    fill="currentColor"
    className={className}
    aria-hidden="true"
  >
    <path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z" />
  </svg>
);

const DrawerList = ({ items }: { items: { to: string; label: string }[] }) => (
  <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
    {items.map(({ to, label }) => (
      <li key={to}>
        <NavLink
          to={to}
          viewTransition
          className={({ isActive }) =>
            isActive ? drawerLinkActive : drawerLink
          }
        >
          {label}
        </NavLink>
      </li>
    ))}
  </ul>
);

function isItem(entry: NavGroupEntry): entry is { to: string; label: string } {
  return "to" in entry;
}

const DrawerEntries = ({ entries }: { entries: NavGroupEntry[] }) => {
  const chunks: { key: string; element: React.ReactNode }[] = [];
  let itemBuf: { to: string; label: string }[] = [];

  const flushItems = () => {
    if (itemBuf.length === 0) return;
    chunks.push({
      key: `items-${itemBuf[0].to}`,
      element: <DrawerList items={itemBuf} />,
    });
    itemBuf = [];
  };

  for (const entry of entries) {
    if (isItem(entry)) {
      itemBuf.push(entry);
    } else {
      flushItems();
      chunks.push({
        key: `sub-${entry.title}`,
        element: (
          <details>
            <summary className={drawerGroupSummary}>
              {entry.title}
              <ChevronIcon className={drawerGroupChevron} />
            </summary>
            <div className={drawerGroupItems}>
              <DrawerList items={entry.items} />
            </div>
          </details>
        ),
      });
    }
  }
  flushItems();

  return (
    <>
      {chunks.map(({ key, element }) => (
        <div key={key}>{element}</div>
      ))}
    </>
  );
};

export const Header = () => {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  return (
    <>
      <header className={headerStyle}>
        <div className={headerInner}>
          <Link to="/" className={logoLink} viewTransition>
            <img
              src="/toro/img/favicon.png"
              alt="Toro"
              width={28}
              height={28}
              className={logoImage}
            />
            <Heading size="3">Toro</Heading>
          </Link>
          <div className={headerRight}>
            <a
              href="https://github.com/toro-ml/toro"
              target="_blank"
              rel="noopener noreferrer"
              className={iconLink}
            >
              <GitHubLogoIcon width="1.5em" height="1.5em" />
            </a>
            <button
              className={menuButton}
              onClick={() => setOpen((v) => !v)}
              aria-label="Menu"
            >
              {open ? (
                <Cross1Icon width="1.25em" height="1.25em" />
              ) : (
                <HamburgerMenuIcon width="1.25em" height="1.25em" />
              )}
            </button>
          </div>
        </div>
      </header>

      <div
        className={backdrop[open ? "open" : "closed"]}
        onClick={() => setOpen(false)}
      />
      <nav className={drawer[open ? "open" : "closed"]}>
        {navSections.map((section, i) => (
          <div key={i}>
            {section.title && (
              <div className={drawerSectionHeading}>{section.title}</div>
            )}
            {section.items && (
              <DrawerList items={section.items} />
            )}
            {section.groups?.map((group) => (
              <details key={group.title}>
                <summary className={drawerGroupSummary}>
                  {group.title}
                  <ChevronIcon className={drawerGroupChevron} />
                </summary>
                <div className={drawerGroupItems}>
                  <DrawerEntries entries={group.entries} />
                </div>
              </details>
            ))}
          </div>
        ))}
      </nav>
    </>
  );
};
