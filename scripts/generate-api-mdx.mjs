import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dirname, "..");
const refDir = join(root, ".fsdocs-out", "reference");
const outDir = join(root, "docs", "app", "content");

const modulePages = {
  "toro-tensor.html": {
    slug: "api-tensor",
    title: "Tensor Module",
    description: "Tensor creation, manipulation, and computation.",
  },
  "toro-device.html": {
    slug: "api-device",
    title: "Device Module",
    description: "Device selection (CPU / CUDA).",
  },
  "toro-dtype.html": {
    slug: "api-dtype",
    title: "DType Module",
    description: "Data type definitions for tensors.",
  },
  "toro-shape.html": {
    slug: "api-shape",
    title: "Shape Module",
    description: "Tensor shape utilities.",
  },
  "toro-nn-linear.html": {
    slug: "api-linear",
    title: "Linear Module",
    description: "Fully connected linear layer.",
  },
  "toro-nn-conv1d.html": {
    slug: "api-conv1d",
    title: "Conv1d Module",
    description: "1D convolutional layer.",
  },
  "toro-nn-conv2d.html": {
    slug: "api-conv2d",
    title: "Conv2d Module",
    description: "2D convolutional layer.",
  },
  "toro-nn-loss.html": {
    slug: "api-loss",
    title: "Loss Module",
    description: "Loss functions for training.",
  },
  "toro-nn-model.html": {
    slug: "api-model",
    title: "Model Module",
    description: "Model composition and parameter management.",
  },
  "toro-tensorop.html": {
    slug: "api-tensorop",
    title: "TensorOp Module",
    description: "Result-returning arithmetic operators.",
  },
  "toro-tensorr.html": {
    slug: "api-tensorr",
    title: "TensorR Module",
    description: "Pipeable Result-returning tensor functions.",
  },
};

function decodeEntities(text) {
  return text
    .replace(/&#32;/g, " ")
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"');
}

function stripTags(html) {
  return decodeEntities(html.replace(/<[^>]+>/g, "")).trim();
}

function stripTagsKeepCode(html) {
  return decodeEntities(
    html
      .replace(/<(?:code|c)>([\s\S]*?)<\/(?:code|c)>/g, "`$1`")
      .replace(/<[^>]+>/g, ""),
  ).trim();
}

function escapeMdx(text) {
  return text.replace(/(\w+)<([^>]+)>/g, "`$1<$2>`");
}

function extractMembers(html) {
  const members = [];
  const tableRegex =
    /<tr>\s*<td class="fsdocs-member-usage">([\s\S]*?)<\/td>\s*<td class="fsdocs-member-xmldoc">([\s\S]*?)<\/td>\s*<\/tr>/g;

  let match;
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

    const params = [];
    const paramRegex =
      /<dt class="fsdocs-param">\s*<span class="fsdocs-param-name">\s*([\s\S]*?)\s*<\/span>\s*:\s*<code>([\s\S]*?)<\/code>\s*<\/dt>/g;
    let paramMatch;
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

function extractTypeMembers(html) {
  const types = [];
  const sectionRegex =
    /<h3>\s*(Union cases|Record fields|Instance members|Static members|Constructors)\s*<\/h3>\s*<table[\s\S]*?<tbody>([\s\S]*?)<\/tbody>/g;

  let match;
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

function extractModuleSummary(html) {
  const match = html.match(
    /<div class="fsdocs-summary-contents">\s*<p class="fsdocs-summary">([\s\S]*?)<\/p>/,
  );
  return match ? stripTagsKeepCode(match[1]) : "";
}

function generateModuleMdx(pageInfo, html) {
  const summary = extractModuleSummary(html);

  const sections = [];
  const sectionRegex =
    /<h3>\s*(Functions and values|Types|Modules)\s*<\/h3>\s*<table[\s\S]*?<tbody>([\s\S]*?)<\/tbody>/g;

  let match;
  while ((match = sectionRegex.exec(html)) !== null) {
    const sectionName = match[1];
    const tbody = match[2];
    const members = extractMembers(tbody);
    if (members.length > 0) {
      sections.push({ title: sectionName, members });
    }
  }

  const typeSections = extractTypeMembers(html);

  let mdx = `---\ntitle: "${pageInfo.title}"\ndescription: "${pageInfo.description}"\n---\n\n`;

  if (summary) {
    mdx += `${escapeMdx(summary)}\n\n`;
  }

  for (const section of typeSections) {
    mdx += `## ${section.section}\n\n`;
    mdx += "| Name | Description |\n";
    mdx += "| --- | --- |\n";
    for (const m of section.members) {
      const sig = escapeMdx(m.signature.replace(/\|/g, "\\|"));
      const desc = escapeMdx(m.summary.replace(/\|/g, "\\|"));
      mdx += `| \`${sig}\` | ${desc} |\n`;
    }
    mdx += "\n";
  }

  for (const section of sections) {
    if (section.title !== "Functions and values") {
      mdx += `## ${section.title}\n\n`;
    }

    mdx += "| Function | Description |\n";
    mdx += "| --- | --- |\n";

    for (const m of section.members) {
      const sig = escapeMdx(m.signature.replace(/\|/g, "\\|"));
      const desc = escapeMdx(m.summary.replace(/\|/g, "\\|"));
      mdx += `| \`${sig}\` | ${desc} |\n`;
    }

    mdx += "\n";

    for (let mi = 0; mi < section.members.length; mi++) {
      const m = section.members[mi];
      if (mi > 0) {
        mdx += `---\n\n`;
      }
      mdx += `### ${escapeMdx(m.name)}\n\n`;
      mdx += "```fsharp\n";
      mdx += m.signature;
      mdx += "\n```\n\n";

      if (m.summary) {
        mdx += `${escapeMdx(m.summary)}\n\n`;
      }

      if (m.params.length > 0) {
        mdx += "**Parameters**\n\n";
        for (const p of m.params) {
          mdx += `- \`${escapeMdx(p.name)}\` : \`${escapeMdx(p.type)}\`\n`;
        }
        mdx += "\n";
      }

      if (m.returns) {
        mdx += `**Returns** \`${escapeMdx(m.returns)}\`\n\n`;
      }
    }
  }

  return mdx;
}

function main() {
  let files;
  try {
    files = readdirSync(refDir);
  } catch {
    console.error(`Reference directory not found: ${refDir}`);
    console.error("Run 'dotnet fsdocs build' first.");
    process.exit(1);
  }

  for (const [filename, pageInfo] of Object.entries(modulePages)) {
    const filePath = join(refDir, filename);
    try {
      const html = readFileSync(filePath, "utf-8");
      const mdx = generateModuleMdx(pageInfo, html);
      const outPath = join(outDir, `${pageInfo.slug}.mdx`);
      writeFileSync(outPath, mdx);
      console.log(`  ${pageInfo.slug}.mdx`);
    } catch (e) {
      console.error(`Error processing ${filename}: ${e.message}`);
    }
  }
}

console.log("Generating API MDX files...");
main();
console.log("Done.");
