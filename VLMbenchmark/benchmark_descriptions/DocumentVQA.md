### Document Visual Question Answering (DocVQA) Benchmark

The goal of **Document Visual Question Answering (DocVQA)** is to evaluate a vision-language model's ability to answer natural-language questions about document images. Unlike **document understanding**, which focuses on interpreting the overall content and structure of a document, DocVQA requires the model to retrieve and reason over the specific information needed to answer a given question. Depending on the task, this information may come from text, tables, charts, forms, or infographics.

A DocVQA benchmark should evaluate:

* **Answer accuracy:** Does the model correctly answer questions using the document content?
* **Document grounding:** Is the answer supported by the relevant text or visual elements within the document?
* **Layout understanding:** Can the model interpret document structure when locating information?
* **Reasoning ability:** Can it combine information across multiple document elements when necessary?

**Recommended benchmark datasets:**

* **DocVQA** – The standard benchmark for question answering over document images.
* **InfographicVQA** – Focuses on answering questions about infographics containing text and visual elements.
* **ChartQA** – Evaluates reasoning over charts and plots.
* **TabFact** and **WikiTableQuestions** – Evaluate reasoning over tabular data.

**Evaluation metrics:**

* Exact Match (EM)
* ANLS (Average Normalized Levenshtein Similarity), commonly used for DocVQA
* F1 score for open-ended question answering

**Example vision-language models capable of DocVQA:**

* Qwen3-VL
* GPT-4o
* LayoutLM (OCR-based baseline)
* Gemma 3

Modern DocVQA systems are often implemented using **multimodal retrieval-augmented generation (RAG)**, where a multimodal retriever first identifies the most relevant document page, and a vision-language model then generates the answer using that page and the user's question. A DocVQA benchmark should therefore assess both accurate document grounding and question answering. Compared with general VQA, DocVQA focuses on visually rich documents rather than natural images and requires understanding document layouts, textual content, and structured visual elements.
