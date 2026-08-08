import { NavLink, useLocation } from "react-router";
import * as s from "./sidebar-style.css";

interface NavItem {
  to: string;
  label: string;
}

interface NavSubGroup {
  title: string;
  items: NavItem[];
}

export type NavGroupEntry = NavItem | NavSubGroup;

interface NavGroup {
  title: string;
  entries: NavGroupEntry[];
}

interface NavSection {
  title: string | null;
  items?: NavItem[];
  groups?: NavGroup[];
}

function isItem(entry: NavGroupEntry): entry is NavItem {
  return "to" in entry;
}

function collectItems(entries: NavGroupEntry[]): NavItem[] {
  return entries.flatMap((e) => (isItem(e) ? [e] : e.items));
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
    groups: [
      {
        title: "Toro",
        entries: [
          { to: "/api-error", label: "Error" },
          { to: "/api-device", label: "Device" },
          { to: "/api-dtype", label: "DType" },
          { to: "/api-shape", label: "Shape" },
          { to: "/api-tensor", label: "Tensor" },
          { to: "/api-tensorop", label: "TensorOp" },
        ],
      },
      {
        title: "Toro.NN",
        entries: [
          { to: "/api-init", label: "Init" },
          { to: "/api-model", label: "Model" },
          {
            title: "Layer",
            items: [
              { to: "/api-linear", label: "Linear" },
              { to: "/api-embedding", label: "Embedding" },
              { to: "/api-conv", label: "Conv" },
              { to: "/api-dropout", label: "Dropout" },
              { to: "/api-layernorm", label: "LayerNorm" },
              { to: "/api-batchnorm", label: "BatchNorm" },
              { to: "/api-groupnorm", label: "GroupNorm" },
              { to: "/api-activation", label: "Activation" },
              { to: "/api-pooling", label: "Pooling" },
            ],
          },
          {
            title: "Block",
            items: [
              { to: "/api-sequential", label: "Sequential" },
              { to: "/api-sequentialt", label: "SequentialT" },
              { to: "/api-func", label: "Func" },
              { to: "/api-rnn", label: "RNN" },
              { to: "/api-kvcache", label: "KvCache" },
              { to: "/api-attention", label: "Attention" },
            ],
          },
          { to: "/api-loss", label: "Loss" },
          { to: "/api-optim", label: "Optim" },
        ],
      },
    ],
  },
];

export const navItems = navSections.flatMap((sec) => [
  ...(sec.items ?? []),
  ...(sec.groups?.flatMap((g) => collectItems(g.entries)) ?? []),
]);

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

const NavList = ({ items }: { items: NavItem[] }) => (
  <ul className={s.list}>
    {items.map(({ to, label }) => (
      <li key={to}>
        <NavLink
          to={to}
          viewTransition
          className={({ isActive }) => (isActive ? s.linkActive : s.link)}
        >
          {label}
        </NavLink>
      </li>
    ))}
  </ul>
);

const GroupEntries = ({
  entries,
  pathname,
}: {
  entries: NavGroupEntry[];
  pathname: string;
}) => {
  const chunks: { key: string; element: React.ReactNode }[] = [];
  let itemBuf: NavItem[] = [];

  const flushItems = () => {
    if (itemBuf.length === 0) return;
    chunks.push({
      key: `items-${itemBuf[0].to}`,
      element: <NavList items={itemBuf} />,
    });
    itemBuf = [];
  };

  for (const entry of entries) {
    if (isItem(entry)) {
      itemBuf.push(entry);
    } else {
      flushItems();
      const active = entry.items.some((item) => item.to === pathname);
      chunks.push({
        key: `sub-${entry.title}`,
        element: (
          <details
            className={s.subGroup}
            open={active || undefined}
          >
            <summary className={s.subGroupSummary}>
              {entry.title}
              <ChevronIcon className={s.groupChevron} />
            </summary>
            <div className={s.subGroupItems}>
              <NavList items={entry.items} />
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

export const Sidebar = () => {
  const { pathname } = useLocation();

  return (
    <nav className={s.nav}>
      {navSections.map((section, i) => (
        <div key={i}>
          {section.title && (
            <div className={s.sectionHeading}>{section.title}</div>
          )}
          {section.items && <NavList items={section.items} />}
          {section.groups?.map((group) => {
            const active = collectItems(group.entries).some(
              (item) => item.to === pathname,
            );
            return (
              <details
                key={group.title}
                className={s.group}
                open={active || undefined}
              >
                <summary className={s.groupSummary}>
                  {group.title}
                  <ChevronIcon className={s.groupChevron} />
                </summary>
                <div className={s.groupItems}>
                  <GroupEntries entries={group.entries} pathname={pathname} />
                </div>
              </details>
            );
          })}
        </div>
      ))}
    </nav>
  );
};
