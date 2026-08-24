import {
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
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
    slug: "api-tensor",
    title: "Tensor APIs",
    description: "Tensor creation, manipulation, and indexing.",
    sources: [
      { html: "toro-tensor.html", heading: "Tensor" },
      { html: "toro-tidx.html", heading: "TIdx" },
      { html: "toro-tensorops.html", heading: "TensorOps" },
      { html: "toro-tensorextensions.html", heading: "TensorExtensions" },
      { html: "toro-tensormodule.html", heading: "Tensor module" },
    ],
  },
  {
    subdir: "toro",
    namespace: "Toro",
    slug: "api-scoped-ownership",
    title: "Scoped ownership",
    description: "Computation expressions for automatic and explicit tensor lifetime management.",
    sources: [
      { html: "toro-scopedbuilder.html", heading: "ScopedBuilder" },
      { html: "toro-explicitscopedbuilder.html", heading: "ExplicitScopedBuilder" },
      { html: "toro-scopedce.html", heading: "ScopedCE" },
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
    description: "Multi-head attention and pre-norm and post-norm transformer blocks.",
    sources: [
      { html: "toro-nn-multiheadattention.html", heading: "MultiHeadAttention" },
      { html: "toro-nn-prenormtransformerblock.html", heading: "PreNormTransformerBlock" },
      { html: "toro-nn-postnormtransformerblock.html", heading: "PostNormTransformerBlock" },
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
    slug: "api-collation",
    title: "Collation",
    description: "Text-to-tensor collation, padding, truncation, and batch encoding.",
    sources: [
      { html: "toro-text-paddingside.html", heading: "PaddingSide" },
      { html: "toro-text-truncationside.html", heading: "TruncationSide" },
      { html: "toro-text-collationlength.html", heading: "CollationLength" },
      { html: "toro-text-collationoptions.html", heading: "CollationOptions" },
      { html: "toro-text-collationoptionsmodule.html", heading: "CollationOptions module" },
      { html: "toro-text-encodedbatch.html", heading: "EncodedBatch" },
      { html: "toro-text-collation.html", heading: "Collation" },
    ],
  },

  // ── Toro.Models ──
  {
    subdir: "toro-models",
    namespace: "Toro.Models",
    slug: "api-causal-lm",
    title: "Causal language models",
    description: "Shared inputs, outputs, contracts, and tensor-level operations for cached causal language models.",
    sources: [
      { html: "toro-models-causallminput-1.html", heading: "CausalLmInput" },
      { html: "toro-models-causallmoutput-1.html", heading: "CausalLmOutput" },
      { html: "toro-models-causallm-1.html", heading: "CausalLm contract" },
      { html: "toro-models-causallm.html", heading: "CausalLm" },
    ],
  },
  {
    subdir: "toro-models",
    namespace: "Toro.Models",
    slug: "api-generation",
    title: "Generation",
    description: "Sampling options and session-based causal language-model generation.",
    sources: [
      { html: "toro-models-generationsampling.html", heading: "GenerationSampling" },
      { html: "toro-models-generationoptions.html", heading: "GenerationOptions" },
      { html: "toro-models-generationoptionsmodule.html", heading: "GenerationOptions module" },
      { html: "toro-models-generationsession-1.html", heading: "GenerationSession" },
      { html: "toro-models-generation.html", heading: "Generation" },
    ],
  },
  {
    subdir: "toro-models",
    namespace: "Toro.Models.Interop",
    slug: "api-model-interop",
    title: "Model interop",
    description: "Shared configuration, model assets, tensor ownership, cache, and causal-input helpers for model-family packages.",
    sources: [
      { html: "toro-models-interop-jsonconfig.html", heading: "JsonConfig" },
      { html: "toro-models-interop-localmodelassets.html", heading: "LocalModelAssets" },
      { html: "toro-models-interop-tensorowner.html", heading: "TensorOwner" },
      { html: "toro-models-interop-fixedkvcache.html", heading: "FixedKvCache" },
      { html: "toro-models-interop-preparedcausalinput.html", heading: "PreparedCausalInput" },
      { html: "toro-models-interop-causalinput.html", heading: "CausalInput" },
    ],
  },

  // ── Toro.Models.SmolLm2 ──
  {
    subdir: "toro-models-smollm2",
    namespace: "Toro.Models",
    slug: "api-smollm2-types",
    title: "SmolLM2 types",
    description: "SmolLM2 configuration and model input and output types.",
    sources: [
      { html: "toro-models-smollm2config.html", heading: "SmolLm2Config" },
      { html: "toro-models-smollm2configmodule.html", heading: "SmolLm2Config module" },
      { html: "toro-models-smollm2input.html", heading: "SmolLm2Input" },
      { html: "toro-models-smollm2output.html", heading: "SmolLm2Output" },
    ],
  },
  {
    subdir: "toro-models-smollm2",
    namespace: "Toro.Models",
    slug: "api-smollm2-cache",
    title: "SmolLM2 cache",
    description: "Key/value cache for incremental SmolLM2 decoding.",
    sources: [{ html: "toro-models-smollm2cache.html", heading: "SmolLm2Cache" }],
  },
  {
    subdir: "toro-models-smollm2",
    namespace: "Toro.Models",
    slug: "api-smollm2-model",
    title: "SmolLM2 model",
    description: "SmolLM2 layers, model composition, local loading, and causal language-model adaptation.",
    sources: [
      { html: "toro-models-smollm2attention.html", heading: "SmolLm2Attention" },
      { html: "toro-models-smollm2mlp.html", heading: "SmolLm2Mlp" },
      { html: "toro-models-smollm2block.html", heading: "SmolLm2Block" },
      { html: "toro-models-smollm2module.html", heading: "SmolLm2" },
    ],
  },

  // ── Toro.Models.DistilGpt2 ──
  {
    subdir: "toro-models-distilgpt2",
    namespace: "Toro.Models",
    slug: "api-distilgpt2-types",
    title: "DistilGPT-2 types",
    description: "DistilGPT-2 configuration and model input and output types.",
    sources: [
      { html: "toro-models-distilgpt2config.html", heading: "DistilGpt2Config" },
      { html: "toro-models-distilgpt2configmodule.html", heading: "DistilGpt2Config module" },
      { html: "toro-models-distilgpt2input.html", heading: "DistilGpt2Input" },
      { html: "toro-models-distilgpt2output.html", heading: "DistilGpt2Output" },
    ],
  },
  {
    subdir: "toro-models-distilgpt2",
    namespace: "Toro.Models",
    slug: "api-distilgpt2-cache",
    title: "DistilGPT-2 cache",
    description: "Key/value cache for incremental DistilGPT-2 decoding.",
    sources: [{ html: "toro-models-distilgpt2cache.html", heading: "DistilGpt2Cache" }],
  },
  {
    subdir: "toro-models-distilgpt2",
    namespace: "Toro.Models",
    slug: "api-distilgpt2-model",
    title: "DistilGPT-2 model",
    description: "DistilGPT-2 layers, model composition, local loading, and causal language-model adaptation.",
    sources: [
      { html: "toro-models-distilgpt2conv1d.html", heading: "DistilGpt2Conv1d" },
      { html: "toro-models-distilgpt2attention.html", heading: "DistilGpt2Attention" },
      { html: "toro-models-distilgpt2mlp.html", heading: "DistilGpt2Mlp" },
      { html: "toro-models-distilgpt2block.html", heading: "DistilGpt2Block" },
      { html: "toro-models-distilgpt2module.html", heading: "DistilGpt2" },
    ],
  },

  // ── Toro.Extensions.AI ──
  {
    subdir: "toro-extensions-ai",
    namespace: "Toro.Extensions.AI",
    slug: "api-causal-lm-chat-client",
    title: "CausalLmChatClient",
    description: "Microsoft.Extensions.AI chat-client adapter for Toro causal language models.",
    sources: [
      { html: "toro-extensions-ai-causallmchatclientconfig-1.html", heading: "CausalLmChatClientConfig" },
      { html: "toro-extensions-ai-causallmchatclient.html", heading: "CausalLmChatClient" },
    ],
  },

  // ── Toro.ML ──
  {
    subdir: "toro-ml",
    namespace: "Toro.ML",
    slug: "api-ranking-dataset",
    title: "RankingDataset",
    description: "Borrowed tensor datasets for learning-to-rank tasks.",
    sources: [
      { html: "toro-ml-rankingdataset.html", heading: "RankingDataset" },
      { html: "toro-ml-rankingdatasetmodule.html", heading: "RankingDataset module" },
    ],
  },
  {
    subdir: "toro-ml",
    namespace: "Toro.ML",
    slug: "api-regression-dataset",
    title: "RegressionDataset",
    description: "Borrowed tensor datasets for regression tasks.",
    sources: [
      { html: "toro-ml-regressiondataset.html", heading: "RegressionDataset" },
      { html: "toro-ml-regressiondatasetmodule.html", heading: "RegressionDataset module" },
    ],
  },
  {
    subdir: "toro-ml",
    namespace: "Toro.ML.Interop",
    slug: "api-ml-interop",
    title: "ML.NET interop",
    description: "Tensor and IDataView conversion used by Toro.ML algorithm packages.",
    sources: [
      { html: "toro-ml-interop-columns.html", heading: "Columns" },
      { html: "toro-ml-interop-regressionrow.html", heading: "RegressionRow" },
      { html: "toro-ml-interop-rankingrow.html", heading: "RankingRow" },
      { html: "toro-ml-interop-scoringrow.html", heading: "ScoringRow" },
      { html: "toro-ml-interop-tensordataview.html", heading: "TensorDataView" },
      { html: "toro-ml-interop-regressiondataview.html", heading: "RegressionDataView" },
      { html: "toro-ml-interop-rankingdataview.html", heading: "RankingDataView" },
    ],
  },

  // ── Toro.ML.Linear ──
  {
    subdir: "toro-ml-linear",
    namespace: "Toro.ML.Linear.Sdca",
    slug: "api-sdca-regression",
    title: "SDCA regression",
    description: "SDCA regression configuration, model persistence, training, prediction, and evaluation.",
    sources: [
      { html: "toro-ml-linear-sdca-regressionconfig.html", heading: "RegressionConfig" },
      { html: "toro-ml-linear-sdca-regressionconfigmodule.html", heading: "RegressionConfig module" },
      { html: "toro-ml-linear-sdca-regressionmodel.html", heading: "RegressionModel" },
      { html: "toro-ml-linear-sdca-regression.html", heading: "Regression" },
    ],
  },

  // ── Toro.ML.FastTree ──
  {
    subdir: "toro-ml-fasttree",
    namespace: "Toro.ML.FastTree",
    slug: "api-fasttree-regression",
    title: "FastTree regression",
    description: "FastTree regression configuration, model persistence, training, prediction, and evaluation.",
    sources: [
      { html: "toro-ml-fasttree-regressionconfig.html", heading: "RegressionConfig" },
      { html: "toro-ml-fasttree-regressionconfigmodule.html", heading: "RegressionConfig module" },
      { html: "toro-ml-fasttree-regressionmodel.html", heading: "RegressionModel" },
      { html: "toro-ml-fasttree-regression.html", heading: "Regression" },
    ],
  },
  {
    subdir: "toro-ml-fasttree",
    namespace: "Toro.ML.FastTree",
    slug: "api-fasttree-ranking",
    title: "FastTree ranking",
    description: "FastTree ranking configuration, model persistence, training, prediction, and evaluation.",
    sources: [
      { html: "toro-ml-fasttree-rankingconfig.html", heading: "RankingConfig" },
      { html: "toro-ml-fasttree-rankingconfigmodule.html", heading: "RankingConfig module" },
      { html: "toro-ml-fasttree-rankingmodel.html", heading: "RankingModel" },
      { html: "toro-ml-fasttree-ranking.html", heading: "Ranking" },
    ],
  },

  // ── Toro.ML.LightGbm ──
  {
    subdir: "toro-ml-lightgbm",
    namespace: "Toro.ML.LightGbm",
    slug: "api-lightgbm-regression",
    title: "LightGBM regression",
    description: "LightGBM regression configuration, model persistence, training, prediction, and evaluation.",
    sources: [
      { html: "toro-ml-lightgbm-regressionconfig.html", heading: "RegressionConfig" },
      { html: "toro-ml-lightgbm-regressionconfigmodule.html", heading: "RegressionConfig module" },
      { html: "toro-ml-lightgbm-regressionmodel.html", heading: "RegressionModel" },
      { html: "toro-ml-lightgbm-regression.html", heading: "Regression" },
    ],
  },
  {
    subdir: "toro-ml-lightgbm",
    namespace: "Toro.ML.LightGbm",
    slug: "api-lightgbm-ranking",
    title: "LightGBM ranking",
    description: "LightGBM ranking configuration, model persistence, training, prediction, and evaluation.",
    sources: [
      { html: "toro-ml-lightgbm-rankingconfig.html", heading: "RankingConfig" },
      { html: "toro-ml-lightgbm-rankingconfigmodule.html", heading: "RankingConfig module" },
      { html: "toro-ml-lightgbm-rankingmodel.html", heading: "RankingModel" },
      { html: "toro-ml-lightgbm-ranking.html", heading: "Ranking" },
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
  return text
    .replace(/(\w+)<([^>]+)>/g, "`$1&lt;$2&gt;`")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeTableCell(text: string): string {
  return text.replace(/(?<!\\)\|/g, "\\|");
}

function escapeInlineCode(text: string): string {
  return text.replace(/</g, "&lt;").replace(/>/g, "&gt;");
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
      const sig = escapeInlineCode(escapeTableCell(m.signature));
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
      const sig = escapeInlineCode(escapeTableCell(m.signature));
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

  return `${mdx.trimEnd()}\n`;
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
    const dir = join(outDir, sub);
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
  }

  let failureCount = 0;

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
      failureCount += 1;
      const htmlList = filePage.sources.map((s) => s.html).join(", ");
      const message = e instanceof Error ? e.message : String(e);
      console.error(`Error processing ${htmlList}: ${message}`);
    }
  }

  if (failureCount > 0) {
    throw new Error(`Failed to generate ${failureCount} API page(s).`);
  }
}

console.log("Generating API MDX files...");
main();
console.log("Done.");
