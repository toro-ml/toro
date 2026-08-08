import { Link } from "react-router";
import type { NavItem } from "~/ui/nav";
import * as s from "./page-nav-style.css";

function setTransitionDirection(direction: "forward" | "back") {
  document.documentElement.dataset.direction = direction;
}

export const PageNav = ({
  previous,
  next,
}: {
  previous: NavItem | null;
  next: NavItem | null;
}) => {
  if (!previous && !next) return null;

  return (
    <nav className={s.root} aria-label="Adjacent pages">
      {previous ? (
        <Link
          to={previous.to}
          className={s.link}
          viewTransition
          onClick={() => setTransitionDirection("back")}
        >
          <span className={s.label}>Previous</span>
          <span className={s.title}>{previous.label}</span>
        </Link>
      ) : (
        <span />
      )}
      {next && (
        <Link
          to={next.to}
          className={s.next}
          viewTransition
          onClick={() => setTransitionDirection("forward")}
        >
          <span className={s.label}>Next</span>
          <span className={s.title}>{next.label}</span>
        </Link>
      )}
    </nav>
  );
};
