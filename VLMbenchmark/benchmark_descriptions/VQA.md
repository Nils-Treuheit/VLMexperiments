### Visual Question Answering (VQA) Benchmark

The goal of **Visual Question Answering (VQA)** is to evaluate a vision-language model's ability to answer natural-language questions about an image. Unlike **image captioning**, which generates a general description of an image, VQA requires the model to interpret both the image and the accompanying question, focusing only on the information relevant to the query. Questions may involve object recognition, counting, localization, comparison, or commonsense reasoning.

A VQA benchmark should evaluate:

* **Answer accuracy:** Does the model correctly answer the question based on the image?
* **Visual grounding:** Is the answer supported by the visual content rather than dataset biases?
* **Reasoning ability:** Can the model perform spatial, logical, and multi-step reasoning when required?
* **Generalization:** Can the model answer questions about previously unseen images and scenarios?

**Recommended benchmark datasets:**

* **VQAv2** – The standard benchmark for general visual question answering.
* **GQA** – Focuses on compositional and relational reasoning.
* **MMMU** – A challenging benchmark covering six academic disciplines with diverse visual inputs such as charts, tables, diagrams, and infographics.
* **MMMU-Pro** – A more rigorous version of MMMU that reduces dataset bias, increases reasoning difficulty, and includes vision-only question settings.

**Evaluation metrics:**

* VQA Accuracy (standard VQA evaluation)
* Exact Match or Multiple-Choice Accuracy (depending on the dataset)
* LLM-based or human evaluation for open-ended responses requiring reasoning

**Example vision-language models capable of VQA:**

* Kimi K2.5
* Qwen3-VL
* Gemma 3
* LLaVA
* BLIP-2
* GPT-4o

A VQA benchmark should compare model answers against ground-truth annotations while assessing not only correctness but also the model's ability to ground its responses in the image and perform visual reasoning. Compared with image captioning, VQA is more targeted and evaluates image understanding conditioned on a specific question rather than producing a general description.
