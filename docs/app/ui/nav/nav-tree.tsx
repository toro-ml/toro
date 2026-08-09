import { useEffect, useState, type ReactNode } from "react";
import { NavLink } from "react-router";
import { classNames } from "~/ui/class-names";
import {
  chunkNavEntries,
  hasActiveNavEntry,
  hasActiveNavItem,
  navSections,
  type NavGroupEntry,
  type NavItem,
} from "./nav-data";
import * as s from "./nav-style.css";

type NavVariant = "sidebar" | "drawer";
type NavLevel = "root" | "group" | "subgroup";

const linkIndent = {
  sidebar: {
    group: s.linkIndent.sidebarGroup,
    subgroup: s.linkIndent.sidebarSubgroup,
  },
  drawer: {
    group: s.linkIndent.drawerGroup,
    subgroup: s.linkIndent.drawerSubgroup,
  },
} as const;

const ChevronIcon = ({
  className,
}: {
  className?: string;
}) => (
  <svg
    viewBox="0 0 16 16"
    fill="currentColor"
    className={className}
    aria-hidden="true"
  >
    <path d="M6.22 3.22a.75.75 0 0 1 1.06 0l4.25 4.25a.75.75 0 0 1 0 1.06l-4.25 4.25a.75.75 0 0 1-1.06-1.06L9.94 8 6.22 4.28a.75.75 0 0 1 0-1.06Z" />
  </svg>
);

function indentFor(variant: NavVariant, level: NavLevel) {
  return level === "root" ? s.linkIndent.none : linkIndent[variant][level];
}

const NavList = ({
  items,
  variant,
  level,
}: {
  items: readonly NavItem[];
  variant: NavVariant;
  level: NavLevel;
}) => (
  <ul className={s.list}>
    {items.map(({ to, label }) => (
      <li key={to}>
        <NavLink
          to={to}
          viewTransition
          className={({ isActive }) =>
            classNames(
              s.link[variant],
              indentFor(variant, level),
              isActive && s.linkActive,
            )
          }
        >
          {label}
        </NavLink>
      </li>
    ))}
  </ul>
);

const NavDetails = ({
  active,
  children,
  className,
  summary,
  summaryClassName,
  variant,
}: {
  active: boolean;
  children: ReactNode;
  className?: string;
  summary: ReactNode;
  summaryClassName: string;
  variant: NavVariant;
}) => {
  const [open, setOpen] = useState(active);

  useEffect(() => {
    if (active) {
      setOpen(true);
    }
  }, [active]);

  if (variant === "drawer") {
    return (
      <details className={className}>
        <summary className={summaryClassName}>{summary}</summary>
        {children}
      </details>
    );
  }

  return (
    <details
      className={className}
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className={summaryClassName}>{summary}</summary>
      {children}
    </details>
  );
};

const GroupEntries = ({
  entries,
  pathname,
  variant,
}: {
  entries: readonly NavGroupEntry[];
  pathname: string;
  variant: NavVariant;
}) =>
  chunkNavEntries(entries).map((chunk) => {
    if (chunk.kind === "items") {
      return (
        <div key={`items-${chunk.items[0].to}`}>
          <NavList items={chunk.items} variant={variant} level="group" />
        </div>
      );
    }

    const { subgroup } = chunk;
    const active = hasActiveNavItem(subgroup.items, pathname);
    return (
      <div key={`sub-${subgroup.title}`}>
        <NavDetails
          active={active}
          summaryClassName={s.subgroupSummary[variant]}
          summary={
            <>
              {subgroup.title}
              <ChevronIcon className={s.chevron[variant]} />
            </>
          }
          variant={variant}
        >
          <div className={s.groupItems[variant]}>
            <NavList
              items={subgroup.items}
              variant={variant}
              level="subgroup"
            />
          </div>
        </NavDetails>
      </div>
    );
  });

export const NavTree = ({
  pathname,
  variant,
}: {
  pathname: string;
  variant: NavVariant;
}) => (
  <>
    {navSections.map((section, sectionIndex) => (
      <div key={section.title ?? sectionIndex}>
        {section.title && (
          <div className={s.sectionHeading[variant]}>{section.title}</div>
        )}
        {section.items && (
          <NavList items={section.items} variant={variant} level="root" />
        )}
        {section.groups?.map((group) => {
          const active = hasActiveNavEntry(group.entries, pathname);
          return (
            <NavDetails
              key={group.title}
              active={active}
              className={s.group[variant]}
              summaryClassName={s.groupSummary[variant]}
              summary={
                <>
                  {group.title}
                  <ChevronIcon className={s.chevron[variant]} />
                </>
              }
              variant={variant}
            >
              <div className={s.groupItems[variant]}>
                <GroupEntries
                  entries={group.entries}
                  pathname={pathname}
                  variant={variant}
                />
              </div>
            </NavDetails>
          );
        })}
      </div>
    ))}
  </>
);
