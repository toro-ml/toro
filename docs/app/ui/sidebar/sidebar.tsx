import { NavLink } from "react-router";
import * as s from "./sidebar-style.css";

const items = [
  { to: "/getting-started", label: "Getting Started" },
  { to: "/concepts", label: "Design" },
  { to: "/tensor", label: "Tensor" },
  { to: "/nn", label: "Neural Networks" },
  { to: "/training", label: "Training" },
];

export const Sidebar = () => (
  <nav className={s.nav}>
    <ul className={s.list}>
      {items.map(({ to, label }) => (
        <li key={to}>
          <NavLink
            to={to}
            className={({ isActive }) => (isActive ? s.linkActive : s.link)}
          >
            {label}
          </NavLink>
        </li>
      ))}
    </ul>
  </nav>
);
