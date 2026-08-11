export interface NavItem {
  to: string;
  label: string;
}

export interface NavSubGroup {
  title: string;
  items: readonly NavItem[];
}

export type NavGroupEntry = NavItem | NavSubGroup;

export type NavEntryChunk =
  | { kind: "items"; items: readonly NavItem[] }
  | { kind: "subgroup"; subgroup: NavSubGroup };

export interface NavGroup {
  title: string;
  entries: readonly NavGroupEntry[];
}

export interface NavSection {
  title: string | null;
  items?: readonly NavItem[];
  groups?: readonly NavGroup[];
}

export function isNavItem(entry: NavGroupEntry): entry is NavItem {
  return "to" in entry;
}

export function collectNavItems(entries: readonly NavGroupEntry[]): NavItem[] {
  return entries.flatMap((entry) =>
    isNavItem(entry) ? [entry] : entry.items,
  );
}

export function chunkNavEntries(
  entries: readonly NavGroupEntry[],
): NavEntryChunk[] {
  return entries.reduce<NavEntryChunk[]>((chunks, entry) => {
    if (!isNavItem(entry)) {
      return [...chunks, { kind: "subgroup", subgroup: entry }];
    }

    const previous = chunks.at(-1);
    if (previous?.kind !== "items") {
      return [...chunks, { kind: "items", items: [entry] }];
    }

    return [
      ...chunks.slice(0, -1),
      { kind: "items", items: [...previous.items, entry] },
    ];
  }, []);
}

export function hasActiveNavItem(
  items: readonly NavItem[],
  pathname: string,
) {
  return items.some(({ to }) => to === pathname);
}

export function hasActiveNavEntry(
  entries: readonly NavGroupEntry[],
  pathname: string,
) {
  return hasActiveNavItem(collectNavItems(entries), pathname);
}

export const navSections: readonly NavSection[] = [
  {
    title: null,
    items: [
      { to: "/getting-started", label: "Getting Started" },
      { to: "/concepts", label: "Core Concepts" },
      { to: "/tensor", label: "Tensor" },
      { to: "/nn", label: "Neural Networks" },
      { to: "/training", label: "Training" },
      { to: "/gnn", label: "Graph Neural Networks" },
      { to: "/vision", label: "Vision" },
      { to: "/text", label: "Text Processing" },
      { to: "/hub", label: "Hub" },
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
          { to: "/api-safetensors", label: "SafeTensors" },
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
              { to: "/api-instancenorm", label: "InstanceNorm" },
              { to: "/api-activation", label: "Activation" },
              { to: "/api-pooling", label: "Pooling" },
            ],
          },
          {
            title: "Block",
            items: [
              { to: "/api-sequential", label: "Sequential" },
              { to: "/api-func", label: "Func" },
              { to: "/api-rnn", label: "RNN" },
              { to: "/api-kvcache", label: "KvCache" },
              { to: "/api-attention", label: "Attention" },
            ],
          },
          { to: "/api-loss", label: "Loss" },
          { to: "/api-metrics", label: "Metrics" },
          { to: "/api-optim", label: "Optim" },
          { to: "/api-scheduler", label: "Scheduler" },
          { to: "/api-clip", label: "Clip" },
        ],
      },
      {
        title: "Toro.GNN",
        entries: [
          {
            title: "Data",
            items: [
              { to: "/api-graphdata", label: "GraphData" },
              { to: "/api-batch", label: "Batch" },
            ],
          },
          { to: "/api-graphutils", label: "GraphUtils" },
          { to: "/api-messagepassing", label: "MessagePassing" },
          {
            title: "Conv",
            items: [
              { to: "/api-gcnconv", label: "GCNConv" },
              { to: "/api-gatconv", label: "GATConv" },
              { to: "/api-sageconv", label: "SAGEConv" },
              { to: "/api-ginconv", label: "GINConv" },
            ],
          },
          { to: "/api-graphnorm", label: "GraphNorm" },
          { to: "/api-globalpool", label: "GlobalPool" },
        ],
      },
      {
        title: "Toro.Hub",
        entries: [
          { to: "/api-hub", label: "Hub" },
        ],
      },
      {
        title: "Toro.Vision",
        entries: [
          { to: "/api-image", label: "Image" },
          { to: "/api-skiatransform", label: "SkiaTransform" },
          { to: "/api-transform", label: "Transform" },
        ],
      },
      {
        title: "Toro.Text",
        entries: [
          { to: "/api-tokenizer", label: "Tokenizer" },
          { to: "/api-encode", label: "Encode" },
        ],
      },
    ],
  },
];

export const navItems = navSections.flatMap((section) => [
  ...(section.items ?? []),
  ...(section.groups?.flatMap((group) => collectNavItems(group.entries)) ?? []),
]);

export function adjacentNavItems(pathname: string) {
  const currentIndex = navItems.findIndex(({ to }) => to === pathname);
  if (currentIndex < 0) {
    return { previous: null, next: null };
  }

  return {
    previous: currentIndex > 0 ? navItems[currentIndex - 1] : null,
    next: navItems.at(currentIndex + 1) ?? null,
  };
}
