### Text Retrieval from Images (OCR) Benchmark

The goal of **text retrieval from images (OCR)** is to evaluate a vision-language model's ability to detect, recognize, and understand textual information embedded within visual content. Unlike **image understanding**, which focuses on interpreting objects, scenes, and visual concepts, OCR requires models to extract **fine-grained textual details** from images, including printed text, handwritten text, scene text, and structured documents.

A text retrieval from images benchmark should evaluate:

* **Text detection:** Can the model identify the location and presence of text within an image?
* **Text recognition:** Can the model accurately transcribe text from different fonts, sizes, orientations, and visual conditions?
* **Document understanding:** Can the model interpret the structure and meaning of text in documents, tables, forms, and layouts?
* **Text grounding:** Can the model associate extracted text with its corresponding visual region?
* **Robustness to visual variations:** Can the model handle noisy, blurry, distorted, multilingual, or low-resolution text?

**Recommended benchmark datasets:**

* **TextVQA** – Evaluates the ability to answer questions using text extracted from real-world images, requiring both OCR and visual reasoning.
* **OCRBench** – A comprehensive benchmark for evaluating OCR capabilities of large vision-language models across text recognition, understanding, and reasoning tasks.
* **DocVQA** – Measures document image understanding, including question answering over scanned documents and structured layouts.
* **FUNSD** – Evaluates understanding of forms by extracting and reasoning over text entities and their relationships.
* **IAM Handwriting Database** – A benchmark for handwritten text recognition.
* **ICDAR Robust Reading Benchmarks** – Standard datasets for scene text detection and recognition.

**Evaluation metrics:**

* **Character Error Rate (CER)** and **Word Error Rate (WER)** for text transcription accuracy
* **Exact Match Accuracy** for text retrieval and question answering tasks
* **ANLS (Average Normalized Levenshtein Similarity)** for document visual question answering
* **Precision, Recall, and F1 score** for text detection and extraction
* **mAP (mean Average Precision)** for text localization and region-level retrieval tasks

**Example vision-language models capable of OCR and text retrieval from images:**

* Qwen3-VL
* Qwen3.5
* GPT-4o
* InternVL
* LLaVA-OneVision
* Molmo2
* Florence-2

Text retrieval from images encompasses several related tasks, including **scene text recognition**, **document visual question answering**, **image-to-text extraction**, and **text-grounded reasoning**. A strong OCR benchmark should therefore evaluate not only whether a model can accurately read text, but also whether it can use extracted text to answer questions, infer relationships, and perform higher-level reasoning over visual documents.

Compared with standard image understanding tasks, OCR introduces additional challenges because the relevant information is often represented by small-scale visual patterns rather than high-level semantic concepts. Effective OCR evaluation should measure both low-level text extraction ability and the model’s capacity to integrate retrieved text with visual context for complex multimodal reasoning.
