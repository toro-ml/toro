import { mkdirSync } from "node:fs";
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dirname, "..");
const refDir = join(root, ".fsdocs-out", "reference");
const outDir = join(root, "docs", "app", "content");

type Source = {
  html: string;
  heading: string;
};

type FilePage = {
  subdir: string;
  namespace: string;
  slug: string;
  title: string;
  description: string;
  sources: Source[];
};

type Parameter = {
  name: string;
  type: string;
};

type Member = {
  name: string;
  signature: string;
  summary: string;
  params: Parameter[];
  returns: string;
};

type MemberSection = {
  section: string;
  members: Member[];
};

type SourceSection = {
  title: string;
  members: Member[];
};

// ── File-based page definitions ──────────────────────────────────────
// Each entry corresponds to a single .fs file in the fsproj.
// `sources` lists the FSDocs HTML files produced by that file.
// Multi-source pages are merged into a single MDX with per-type h2 headings.

const filePages: FilePage[] = [
  // ── Toro ── (fsproj compile order)
  {
    subdir: "toro",
    namespace: "Toro",
    slug: "api-device",
    title: "Device",
    description: "Device selection (CPU / CUDA).",
    sources: [{ html: "toro-device.html", heading: "Device" }],
  },
  {
    subdir: "toro",
    namespace: "Toro",
    slug: "api-dtype",
    title: "DType",
    description: "Data type definitions for tensors.",
    sources: [{ html: "toro-dtype.html", heading: "DType" }],
  },
  {
    subdir: "toro",
    namespace: "Toro",
    slug: "api-shape",
    title: "Shape",
    description: "Tensor shape utilities.",
    sources: [{ html: "toro-shape.html", heading: "Shape" }],
  },
  {
    subdir: "toro",
    namespace: "Toro",
    slug: "api-tensor",
    title: "Tensor APIs",
    description: "Tensor creation, manipulation, and indexing.",
    sources: [
      { html: "toro-tensor.html", heading: "Tensor" },
      { html: "toro-tidx.html", heading: "TIdx" },
    ],
  },
  // ── Toro.NN ── (fsproj compile order)
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-init",
    title: "Init",
    description: "Weight initialization strategies.",
    sources: [{ html: "toro-nn-init.html", heading: "Init" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-model",
    title: "Model",
    description: "Model composition and parameter management.",
    sources: [{ html: "toro-nn-model.html", heading: "Model" }],
  },
  // Layer/
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-linear",
    title: "Linear",
    description: "Fully connected linear layer.",
    sources: [{ html: "toro-nn-linear.html", heading: "Linear" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-embedding",
    title: "Embedding",
    description: "Embedding lookup layer.",
    sources: [{ html: "toro-nn-embedding.html", heading: "Embedding" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-conv",
    title: "Convolution layers",
    description: "Convolutional and transposed convolutional layers (1-D and 2-D).",
    sources: [
      { html: "toro-nn-conv1d.html", heading: "Conv1d" },
      { html: "toro-nn-conv2d.html", heading: "Conv2d" },
      { html: "toro-nn-convtranspose1d.html", heading: "ConvTranspose1d" },
      { html: "toro-nn-convtranspose2d.html", heading: "ConvTranspose2d" },
    ],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-dropout",
    title: "Dropout",
    description: "Dropout regularization layer.",
    sources: [{ html: "toro-nn-dropout.html", heading: "Dropout" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-layernorm",
    title: "Normalization layers",
    description: "Layer normalization and RMS normalization.",
    sources: [
      { html: "toro-nn-layernorm.html", heading: "LayerNorm" },
      { html: "toro-nn-rmsnorm.html", heading: "RmsNorm" },
    ],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-batchnorm",
    title: "BatchNorm",
    description: "Batch normalization.",
    sources: [{ html: "toro-nn-batchnorm.html", heading: "BatchNorm" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-groupnorm",
    title: "GroupNorm",
    description: "Group normalization.",
    sources: [{ html: "toro-nn-groupnorm.html", heading: "GroupNorm" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-instancenorm",
    title: "InstanceNorm",
    description: "Instance normalization.",
    sources: [{ html: "toro-nn-instancenorm.html", heading: "InstanceNorm" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-activation",
    title: "Activation",
    description: "Activation functions as module wrappers.",
    sources: [{ html: "toro-nn-activation.html", heading: "Activation" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-pooling",
    title: "Pooling layers",
    description: "Pooling layers (max, average, and adaptive).",
    sources: [
      { html: "toro-nn-maxpool1d.html", heading: "MaxPool1d" },
      { html: "toro-nn-maxpool2d.html", heading: "MaxPool2d" },
      { html: "toro-nn-avgpool1d.html", heading: "AvgPool1d" },
      { html: "toro-nn-avgpool2d.html", heading: "AvgPool2d" },
      { html: "toro-nn-adaptiveavgpool1d.html", heading: "AdaptiveAvgPool1d" },
      { html: "toro-nn-adaptiveavgpool2d.html", heading: "AdaptiveAvgPool2d" },
    ],
  },
  // Block/
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-sequential",
    title: "Sequential",
    description: "Sequential module composition.",
    sources: [{ html: "toro-nn-sequential.html", heading: "Sequential" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-func",
    title: "Func",
    description: "Custom function as a module.",
    sources: [{ html: "toro-nn-func.html", heading: "Func" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-rnn",
    title: "Recurrent layers",
    description: "Recurrent layers (LSTM and GRU).",
    sources: [
      { html: "toro-nn-lstm.html", heading: "LSTM" },
      { html: "toro-nn-gru.html", heading: "GRU" },
    ],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-kvcache",
    title: "KvCache",
    description: "Key-value cache for inference.",
    sources: [{ html: "toro-nn-kvcache.html", heading: "KvCache" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-attention",
    title: "Attention layers",
    description: "Multi-head attention and transformer block.",
    sources: [
      { html: "toro-nn-multiheadattention.html", heading: "MultiHeadAttention" },
      { html: "toro-nn-transformerblock.html", heading: "TransformerBlock" },
    ],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-loss",
    title: "Loss",
    description: "Loss functions for training.",
    sources: [{ html: "toro-nn-loss.html", heading: "Loss" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-optim",
    title: "Optimizers",
    description: "Optimizers (SGD and AdamW).",
    sources: [
      { html: "toro-nn-sgd.html", heading: "SGD" },
      { html: "toro-nn-adamw.html", heading: "AdamW" },
    ],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-scheduler",
    title: "Scheduler",
    description: "Learning rate schedulers.",
    sources: [
      { html: "toro-nn-lrschedule.html", heading: "LrSchedule" },
      { html: "toro-nn-scheduler.html", heading: "Scheduler" },
    ],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-clip",
    title: "Clip",
    description: "Gradient clipping utilities.",
    sources: [{ html: "toro-nn-clip.html", heading: "Clip" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-metrics",
    title: "Metrics",
    description: "Evaluation metrics (accuracy, precision, recall, F1).",
    sources: [{ html: "toro-nn-metrics.html", heading: "Metrics" }],
  },
  {
    subdir: "toro-nn",
    namespace: "Toro.NN",
    slug: "api-checkpoint",
    title: "Checkpoint",
    description: "Save and restore full training state (model + optimizer + epoch).",
    sources: [{ html: "toro-nn-checkpoint.html", heading: "Checkpoint" }],
  },

  {
    subdir: "toro",
    namespace: "Toro",
    slug: "api-safetensors",
    title: "SafeTensors",
    description: "Read and write the SafeTensors binary format.",
    sources: [{ html: "toro-safetensors.html", heading: "SafeTensors" }],
  },

  // ── Toro.Hub ──
  {
    subdir: "toro-hub",
    namespace: "Toro.Hub",
    slug: "api-hub",
    title: "Hub",
    description: "Download files from the Hugging Face Hub.",
    sources: [{ html: "toro-hub-hub.html", heading: "Hub" }],
  },

  // ── Toro.GNN ──
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-graphdata",
    title: "GraphData",
    description: "Graph data representation in COO format.",
    sources: [{ html: "toro-gnn-graphdata.html", heading: "GraphData" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-batch",
    title: "Batch",
    description: "Graph batching utilities.",
    sources: [{ html: "toro-gnn-batch.html", heading: "Batch" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-graphutils",
    title: "GraphUtils",
    description: "Graph utility functions.",
    sources: [{ html: "toro-gnn-graphutils.html", heading: "GraphUtils" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-messagepassing",
    title: "MessagePassing",
    description: "Message passing operations for GNNs.",
    sources: [
      { html: "toro-gnn-messagepassing.html", heading: "MessagePassing" },
    ],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-gcnconv",
    title: "GCNConv",
    description: "Graph Convolutional Network layer.",
    sources: [{ html: "toro-gnn-gcnconv.html", heading: "GCNConv" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-gatconv",
    title: "GATConv",
    description: "Graph Attention Network layer.",
    sources: [{ html: "toro-gnn-gatconv.html", heading: "GATConv" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-sageconv",
    title: "SAGEConv",
    description: "GraphSAGE layer.",
    sources: [{ html: "toro-gnn-sageconv.html", heading: "SAGEConv" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-ginconv",
    title: "GINConv",
    description: "Graph Isomorphism Network layer.",
    sources: [{ html: "toro-gnn-ginconv.html", heading: "GINConv" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-graphnorm",
    title: "GraphNorm",
    description: "Graph normalization layer.",
    sources: [{ html: "toro-gnn-graphnorm.html", heading: "GraphNorm" }],
  },
  {
    subdir: "toro-gnn",
    namespace: "Toro.GNN",
    slug: "api-globalpool",
    title: "GlobalPool",
    description: "Global graph pooling operations.",
    sources: [{ html: "toro-gnn-globalpool.html", heading: "GlobalPool" }],
  },

  // ── Toro.Vision ──
  {
    subdir: "toro-vision",
    namespace: "Toro.Vision",
    slug: "api-image",
    title: "Image",
    description: "Image I/O and SKBitmap-Tensor conversion via SkiaSharp.",
    sources: [
      { html: "toro-vision-imageformat.html", heading: "ImageFormat" },
      { html: "toro-vision-image.html", heading: "Image" },
    ],
  },
  {
    subdir: "toro-vision",
    namespace: "Toro.Vision",
    slug: "api-skiatransform",
    title: "SkiaTransform",
    description: "Spatial transforms operating directly on SKBitmap.",
    sources: [
      { html: "toro-vision-skiatransform.html", heading: "SkiaTransform" },
    ],
  },
  {
    subdir: "toro-vision",
    namespace: "Toro.Vision",
    slug: "api-transform",
    title: "Transform",
    description: "Image transforms for preprocessing and data augmentation.",
    sources: [
      { html: "toro-vision-itransform.html", heading: "ITransform" },
      { html: "toro-vision-compose.html", heading: "Compose" },
      { html: "toro-vision-normalize.html", heading: "Normalize" },
      { html: "toro-vision-normalizemodule.html", heading: "Normalize module" },
      { html: "toro-vision-resize.html", heading: "Resize" },
      { html: "toro-vision-resizemodule.html", heading: "Resize module" },
      { html: "toro-vision-randomhorizontalflip.html", heading: "RandomHorizontalFlip" },
      { html: "toro-vision-randomhorizontalflipmodule.html", heading: "RandomHorizontalFlip module" },
      { html: "toro-vision-randomverticalflip.html", heading: "RandomVerticalFlip" },
      { html: "toro-vision-randomverticalflipmodule.html", heading: "RandomVerticalFlip module" },
      { html: "toro-vision-randomcrop.html", heading: "RandomCrop" },
      { html: "toro-vision-randomcropmodule.html", heading: "RandomCrop module" },
      { html: "toro-vision-centercrop.html", heading: "CenterCrop" },
      { html: "toro-vision-centercropmodule.html", heading: "CenterCrop module" },
      { html: "toro-vision-tograyscale.html", heading: "ToGrayscale" },
      { html: "toro-vision-tograyscalemodule.html", heading: "ToGrayscale module" },
      { html: "toro-vision-convertimagedtype.html", heading: "ConvertImageDType" },
      { html: "toro-vision-convertimagedtypemodule.html", heading: "ConvertImageDType module" },
    ],
  },

  // ── Toro.Text ──
  {
    subdir: "toro-text",
    namespace: "Toro.Text",
    slug: "api-tokenizer",
    title: "Tokenizer",
    description: "Tokenizer wrapping Microsoft.ML.Tokenizers with F#-idiomatic API.",
    sources: [
      { html: "toro-text-tokenizermodule.html", heading: "Tokenizer" },
    ],
  },
  {
    subdir: "toro-text",
    namespace: "Toro.Text",
    slug: "api-encode",
    title: "Encode",
    description: "Text-to-tensor encoding utilities.",
    sources: [
      { html: "toro-text-encode.html", heading: "Encode" },
    ],
  },
];

// ── HTML helpers ──────────────────────────────────────────────────────

function decodeEntities(text: string): string {
  return text
    .replace(/&#32;/g, " ")
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"');
}

function stripTags(html: string): string {
  return decodeEntities(html.replace(/<[^>]+>/g, "")).trim();
}

function stripTagsKeepCode(html: string): string {
  return decodeEntities(
    html
      .replace(/<(?:code|c)>([\s\S]*?)<\/(?:code|c)>/g, "`$1`")
      .replace(/<[^>]+>/g, ""),
  ).trim();
}

function escapeMdx(text: string): string {
  return text.replace(/(\w+)<([^>]+)>/g, "`$1<$2>`");
}

function escapeTableCell(text: string): string {
  return text.replace(/(?<!\\)\|/g, "\\|");
}

function extractMembers(html: string): Member[] {
  const members: Member[] = [];
  const tableRegex =
    /<tr>\s*<td class="fsdocs-member-usage">([\s\S]*?)<\/td>\s*<td class="fsdocs-member-xmldoc">([\s\S]*?)<\/td>\s*<\/tr>/g;

  let match: RegExpExecArray | null;
  while ((match = tableRegex.exec(html)) !== null) {
    const usageHtml = match[1];
    const docHtml = match[2];

    const idMatch = usageHtml.match(/<a id="([^"]+)">/);
    const name = idMatch ? idMatch[1] : "";

    const sigMatch = usageHtml.match(
      /<a href="#[^"]*">\s*<code>([\s\S]*?)<\/code>\s*<\/a>/,
    );
    const signature = sigMatch ? stripTags(sigMatch[1]) : name;

    const summaryMatch = docHtml.match(
      /<p class="fsdocs-summary">([\s\S]*?)<\/p>/,
    );
    const summary = summaryMatch ? stripTagsKeepCode(summaryMatch[1]) : "";

    const params: Parameter[] = [];
    const paramRegex =
      /<dt class="fsdocs-param">\s*<span class="fsdocs-param-name">\s*([\s\S]*?)\s*<\/span>\s*:\s*<code>([\s\S]*?)<\/code>\s*<\/dt>/g;
    let paramMatch: RegExpExecArray | null;
    while ((paramMatch = paramRegex.exec(docHtml)) !== null) {
      params.push({
        name: stripTags(paramMatch[1]),
        type: stripTags(paramMatch[2]),
      });
    }

    const returnMatch = docHtml.match(
      /<span class="fsdocs-return-name">[\s\S]*?<\/span>\s*<code>([\s\S]*?)<\/code>/,
    );
    const returns = returnMatch ? stripTags(returnMatch[1]) : "";

    if (name) {
      members.push({ name, signature, summary, params, returns });
    }
  }
  return members;
}

function extractTypeMembers(html: string): MemberSection[] {
  const types: MemberSection[] = [];
  const sectionRegex =
    /<h3>\s*(Union cases|Record fields|Instance members|Static members|Constructors)\s*<\/h3>\s*<table[\s\S]*?<tbody>([\s\S]*?)<\/tbody>/g;

  let match: RegExpExecArray | null;
  while ((match = sectionRegex.exec(html)) !== null) {
    const sectionName = match[1];
    const tbody = match[2];
    const members = extractMembers(
      `<table><tbody>${tbody}</tbody></table>`,
    );
    if (members.length > 0) {
      types.push({ section: sectionName, members });
    }
  }
  return types;
}

function extractModuleSummary(html: string): string {
  const match = html.match(
    /<div class="fsdocs-summary-contents">\s*<p class="fsdocs-summary">([\s\S]*?)<\/p>/,
  );
  return match ? stripTagsKeepCode(match[1]) : "";
}

// ── MDX generation ───────────────────────────────────────────────────

function h(level: number): string {
  return "#".repeat(level);
}

function generateSourceContent(html: string, sourceLevel: number): string {
  const summary = extractModuleSummary(html);

  const sections: SourceSection[] = [];
  const sectionRegex =
    /<h3>\s*(Functions and values|Types|Modules)\s*<\/h3>\s*<table[\s\S]*?<tbody>([\s\S]*?)<\/tbody>/g;

  let match: RegExpExecArray | null;
  while ((match = sectionRegex.exec(html)) !== null) {
    const sectionName = match[1];
    const tbody = match[2];
    const members = extractMembers(tbody);
    if (members.length > 0) {
      sections.push({ title: sectionName, members });
    }
  }

  const typeSections = extractTypeMembers(html);

  let mdx = "";
  const sectionLevel = sourceLevel + 1;
  const memberLevel = sectionLevel + 1;

  if (summary) {
    mdx += `${escapeMdx(summary)}\n\n`;
  }

  for (const section of typeSections) {
    mdx += `${h(sectionLevel)} ${section.section}\n\n`;
    mdx += "| Name | Description |\n";
    mdx += "| --- | --- |\n";
    for (const m of section.members) {
      const sig = escapeMdx(escapeTableCell(m.signature));
      const desc = escapeMdx(escapeTableCell(m.summary));
      mdx += `| \`${sig}\` | ${desc} |\n`;
    }
    mdx += "\n";
  }

  for (const section of sections) {
    if (section.title !== "Functions and values") {
      mdx += `${h(sectionLevel)} ${section.title}\n\n`;
    }

    mdx += "| Function | Description |\n";
    mdx += "| --- | --- |\n";

    for (const m of section.members) {
      const sig = escapeMdx(escapeTableCell(m.signature));
      const desc = escapeMdx(escapeTableCell(m.summary));
      mdx += `| \`${sig}\` | ${desc} |\n`;
    }

    mdx += "\n";

    for (let mi = 0; mi < section.members.length; mi++) {
      const m = section.members[mi];
      if (mi > 0) {
        mdx += `---\n\n`;
      }
      const level =
        section.title === "Functions and values"
          ? sectionLevel
          : memberLevel;
      mdx += `${h(level)} ${escapeMdx(m.name)}\n\n`;
      mdx += "```fsharp\n";
      mdx += m.signature;
      mdx += "\n```\n\n";

      if (m.summary) {
        mdx += `${escapeMdx(m.summary)}\n\n`;
      }

      if (m.params.length > 0) {
        mdx += "**Parameters**\n\n";
        for (const p of m.params) {
          mdx += `- \`${p.name}\` : \`${p.type}\`\n`;
        }
        mdx += "\n";
      }

      if (m.returns) {
        mdx += `**Returns** \`${m.returns}\`\n\n`;
      }
    }
  }

  return mdx;
}

function generateFileMdx(filePage: FilePage): string {
  const ns = filePage.namespace
    ? `\nnamespace: "${filePage.namespace}"`
    : "";
  const multi = filePage.sources.length > 1;
  const title = multi ? filePage.title : filePage.sources[0].heading;
  const showTitle = multi ? "\nshowTitle: false" : "";
  let mdx = `---\ntitle: "${title}"\ndescription: "${filePage.description}"${ns}${showTitle}\n---\n\n`;

  if (multi) {
    for (let i = 0; i < filePage.sources.length; i++) {
      const source = filePage.sources[i];
      const filePath = join(refDir, source.html);
      const html = readFileSync(filePath, "utf-8");

      if (i > 0) mdx += "---\n\n";
      mdx += `# ${source.heading}\n\n`;
      mdx += generateSourceContent(html, 1);
    }
  } else {
    const source = filePage.sources[0];
    const filePath = join(refDir, source.html);
    const html = readFileSync(filePath, "utf-8");
    mdx += generateSourceContent(html, 1);
  }

  return mdx;
}

// ── Main ─────────────────────────────────────────────────────────────

function main(): void {
  try {
    readdirSync(refDir);
  } catch {
    console.error(`Reference directory not found: ${refDir}`);
    console.error("Run 'dotnet fsdocs build' first.");
    process.exit(1);
  }

  const subdirs = new Set(
    filePages.map((p) => p.subdir).filter(Boolean),
  );
  for (const sub of subdirs) {
    mkdirSync(join(outDir, sub), { recursive: true });
  }

  for (const filePage of filePages) {
    try {
      const mdx = generateFileMdx(filePage);
      const dir = filePage.subdir ? join(outDir, filePage.subdir) : outDir;
      const outPath = join(dir, `${filePage.slug}.mdx`);
      writeFileSync(outPath, mdx);
      const label = filePage.subdir
        ? `${filePage.subdir}/${filePage.slug}.mdx`
        : `${filePage.slug}.mdx`;
      console.log(`  ${label}`);
    } catch (e: unknown) {
      const htmlList = filePage.sources.map((s) => s.html).join(", ");
      const message = e instanceof Error ? e.message : String(e);
      console.error(`Error processing ${htmlList}: ${message}`);
    }
  }
}

console.log("Generating API MDX files...");
main();
console.log("Done.");
