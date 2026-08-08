import {
  Cross1Icon,
  GitHubLogoIcon,
  HamburgerMenuIcon,
} from "@radix-ui/react-icons";
import { Heading } from "@radix-ui/themes";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, NavLink, useLocation } from "react-router";
import { navItems } from "~/ui/sidebar";
import { link, linkActive } from "~/ui/sidebar/sidebar-style.css";
import {
  backdrop,
  drawer,
  headerInner,
  headerRight,
  headerStyle,
  iconLink,
  logoImage,
  logoLink,
  menuButton,
} from "./header-style.css";

function useScrollDirection() {
  const [visible, setVisible] = useState(true);
  const lastY = useRef(0);

  const onScroll = useCallback(() => {
    const y = window.scrollY;
    setVisible(y <= 0 || y < lastY.current);
    lastY.current = y;
  }, []);

  useEffect(() => {
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [onScroll]);

  return visible;
}

export const Header = () => {
  const visible = useScrollDirection();
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  return (
    <>
      <header className={headerStyle[visible || open ? "visible" : "hidden"]}>
        <div className={headerInner}>
          <Link to="/" className={logoLink}>
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
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {navItems.map(({ to, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                className={({ isActive }) => (isActive ? linkActive : link)}
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </>
  );
};
