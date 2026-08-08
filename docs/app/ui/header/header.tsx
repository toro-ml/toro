import { GitHubLogoIcon } from "@radix-ui/react-icons";
import { Heading } from "@radix-ui/themes";
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import {
  headerInner,
  headerStyle,
  iconLink,
  logoImage,
  logoLink,
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
  return (
    <header className={headerStyle[visible ? "visible" : "hidden"]}>
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
        <a
          href="https://github.com/toro-ml/toro"
          target="_blank"
          rel="noopener noreferrer"
          className={iconLink}
        >
          <GitHubLogoIcon width="1.5em" height="1.5em" />
        </a>
      </div>
    </header>
  );
};
