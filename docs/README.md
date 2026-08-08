# Toro Documentation Site

Built with [React Router v7](https://reactrouter.com/) (SPA mode, static pre-rendering).

## Development

```bash
cd docs
pnpm install
pnpm dev
```

## Build

```bash
pnpm build
```

Output goes to `build/client/`. Deployed to GitHub Pages via `.github/workflows/docs.yml`.

## Stack

- **React Router v7** -- framework (SPA + static pre-rendering)
- **Radix UI Themes** -- component library
- **Vanilla Extract** -- type-safe CSS (`*.css.ts`)
- **MDX** -- content authoring (`app/content/*.mdx`)
- **Shiki** -- syntax highlighting for code blocks

## Adding Pages

1. Add an `.mdx` file to `app/content/`.
2. Add a route entry to `app/routes.ts`.
3. Add the slug to `prerender` in `react-router.config.ts`.
4. Add a sidebar link in `app/ui/sidebar/sidebar.tsx`.
