### Document Understanding Benchmark

The goal of **document understanding** is to evaluate a vision-language model's ability to extract, interpret, and reason over information contained in documents. Unlike **visual language retrieval**, which identifies the most relevant document, document understanding focuses on interpreting the retrieved document's content, including text, tables, charts, diagrams, forms, and layout.

A document understanding benchmark should evaluate:

* **Text recognition:** Can the model accurately read printed and handwritten text?
* **Layout understanding:** Can it identify document structure, such as headings, paragraphs, tables, and figures?
* **Multimodal comprehension:** Can it combine textual and visual information to answer questions or summarize content?
* **Information extraction:** Can it accurately extract structured information from complex documents?

**Recommended benchmark datasets:**

* **InfoVQA** – Evaluates question answering over infographics and visually rich documents.
* **OmniDocBench** – A comprehensive benchmark for OCR and document understanding.
* **OlmOCRBench** – Evaluates OCR performance on challenging document images.
* **DocVQA** – A standard benchmark for question answering over document images.

**Evaluation metrics:**

* OCR accuracy (e.g., Character Error Rate (CER), Word Error Rate (WER))
* Exact Match (EM) and F1 score for document question answering
* Task-specific accuracy for information extraction and document comprehension

**Example vision-language models capable of document understanding:**

* GLM-OCR
* OlmOCR 2
* LightOnOCR-2-1B
* Qwen3-VL
* GPT-4o

A document understanding benchmark should assess a model's ability to recognize document content, interpret layout and structure, and integrate textual and visual information to perform downstream tasks such as question answering, summarization, and information extraction. Compared with visual language retrieval, document understanding emphasizes **content interpretation** rather than document search, while extending beyond traditional OCR by incorporating semantic and structural reasoning over complex documents.
