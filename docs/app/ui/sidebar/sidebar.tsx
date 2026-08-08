import { NavLink } from "react-router";
import * as s from "./sidebar-style.css";

interface NavItem {
  to: string;
  label: string;
}

interface NavSection {
  title: string | null;
  items: NavItem[];
}

export const navSections: NavSection[] = [
  {
    title: null,
    items: [
      { to: "/getting-started", label: "Getting Started" },
      { to: "/concepts", label: "Design" },
      { to: "/tensor", label: "Tensor" },
      { to: "/nn", label: "Neural Networks" },
      { to: "/training", label: "Training" },
    ],
  },
  {
    title: "API Reference",
    items: [
      { to: "/api-tensor", label: "Tensor" },
      { to: "/api-device", label: "Device" },
      { to: "/api-dtype", label: "DType" },
      { to: "/api-shape", label: "Shape" },
      { to: "/api-linear", label: "Linear" },
      { to: "/api-conv1d", label: "Conv1d" },
      { to: "/api-conv2d", label: "Conv2d" },
      { to: "/api-tensorop", label: "TensorOp" },
      { to: "/api-tensorr", label: "TensorR" },
      { to: "/api-loss", label: "Loss" },
      { to: "/api-model", label: "Model" },
    ],
  },
];

export const navItems = navSections.flatMap((section) => section.items);

export const Sidebar = () => (
  <nav className={s.nav}>
    {navSections.map((section, i) => (
      <div key={i}>
        {section.title && (
          <div className={s.sectionHeading}>{section.title}</div>
        )}
        <ul className={s.list}>
          {section.items.map(({ to, label }) => (
            <li key={to}>
              <NavLink
                to={to}
                viewTransition
                className={({ isActive }) =>
                  isActive ? s.linkActive : s.link
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
);
