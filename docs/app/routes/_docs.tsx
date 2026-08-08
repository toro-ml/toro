import { Outlet } from "react-router";
import { docsGrid } from "~/ui/layout/docs-layout.css";
import { Sidebar } from "~/ui/sidebar";

export default function DocsLayout() {
  return (
    <div className={docsGrid}>
      <Sidebar />
      <Outlet />
    </div>
  );
}
