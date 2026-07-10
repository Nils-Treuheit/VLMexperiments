### Visual Language Retrieval Benchmark

The goal of **visual language retrieval** is to evaluate a vision-language model's ability to retrieve the most relevant images, documents, or text passages for a given query. Unlike **image captioning**, **VQA**, or **visual reasoning**, retrieval does not generate new content. Instead, it ranks existing items by semantic similarity within a shared image-text embedding space.

Visual language retrieval encompasses three main tasks:

* **Text-to-image retrieval:** Retrieve the most relevant images for a textual query.
* **Image-to-text retrieval:** Retrieve the most relevant textual descriptions or documents for an image query.
* **Image-to-image retrieval:** Retrieve visually or semantically similar images.

Modern retrieval systems also support **visual document retrieval**, where entire document pages containing text, tables, charts, and diagrams are retrieved directly without relying on OCR.

A retrieval benchmark should evaluate:

* **Retrieval accuracy:** Are the most relevant items ranked at the top?
* **Cross-modal alignment:** Does the shared embedding space correctly capture semantic similarity between images and text?
* **Ranking quality:** Are highly relevant results consistently prioritized over less relevant ones?
* **Generalization:** Can the model retrieve relevant content across diverse domains and document types?

**Recommended benchmark datasets:**

* **ViDoRe V3** – The primary benchmark for visual document retrieval, covering multiple professional domains and languages.
* **MMEB-V2** – A benchmark for evaluating multimodal embedding models.
* **MS COCO Retrieval** – A standard benchmark for image-text retrieval.
* **Flickr30k Retrieval** – Commonly used for evaluating cross-modal retrieval performance.

**Evaluation metrics:**

* **NDCG@k (Normalized Discounted Cumulative Gain)** – The primary metric for retrieval quality, measuring how well relevant results are ranked near the top.
* Recall@k
* Precision@k
* Mean Reciprocal Rank (MRR)

**Example vision-language retrieval models:**

* CLIP
* ColPali
* ColQwen2
* Qwen3-VL-Embedding

A visual language retrieval benchmark should measure how effectively a model aligns visual and textual representations to retrieve the most relevant results. Compared with other vision-language tasks, retrieval focuses on **ranking and semantic matching** rather than generating captions, answering questions, or performing multi-step reasoning.
