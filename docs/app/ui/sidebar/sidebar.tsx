import { useLocation } from "react-router";
import { NavTree } from "~/ui/nav";
import { nav } from "./sidebar-style.css";

export const Sidebar = () => {
  const { pathname } = useLocation();

  return (
    <nav className={nav} aria-label="Documentation">
      <NavTree pathname={pathname} variant="sidebar" />
    </nav>
  );
};
