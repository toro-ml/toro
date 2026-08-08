import {
  Cross1Icon,
  GitHubLogoIcon,
  HamburgerMenuIcon,
} from "@radix-ui/react-icons";
import { Heading } from "@radix-ui/themes";
import { useEffect, useState } from "react";
import { Link, NavLink, useLocation } from "react-router";
import { navSections } from "~/ui/sidebar";
import {
  backdrop,
  drawer,
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

export const Header = () => {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

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
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {section.items.map(({ to, label }) => (
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
          </div>
        ))}
      </nav>
    </>
  );
};
