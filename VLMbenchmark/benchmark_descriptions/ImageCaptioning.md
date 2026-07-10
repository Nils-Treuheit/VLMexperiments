### Image Captioning Benchmark

When evaluating **image captioning**, the goal is to measure a vision-language model's ability to generate a coherent and accurate natural-language description of an image. Unlike **Visual Question Answering (VQA)**, where the model answers a user-specified question about an image, image captioning requires the model to independently identify the most relevant objects, attributes, actions, and relationships and summarize them in a descriptive caption.

A benchmark for image captioning should evaluate:

* **Caption accuracy:** Does the caption correctly describe the image content?
* **Completeness:** Are the main objects, actions, and relationships included?
* **Fluency:** Is the caption grammatically correct and natural?
* **Relevance:** Does the caption focus on the important aspects of the scene without hallucinating details?

**Recommended benchmark datasets:**

* **MS COCO Captions** – The standard benchmark containing over 120,000 images with five human-written captions per image.
* **Flickr30k** – Approximately 31,000 images with five reference captions each, commonly used for caption generation evaluation.
* **NoCaps** – Evaluates generalization to novel object categories not seen during training.
* **TextCaps** – Focuses on images containing text, requiring models to incorporate OCR information into captions.

**Evaluation metrics:**

* BLEU
* METEOR
* ROUGE-L
* CIDEr (the most widely used metric for image captioning)
* SPICE
* Additionally, LLM-based or human evaluation can assess semantic correctness and caption quality beyond lexical overlap.

**Example vision-language models capable of image captioning:**

* Gemma 3
* Qwen3.5-VL
* LLaVA
* BLIP-2
* InstructBLIP
* GPT-4o

The benchmark should compare generated captions against reference captions using both automatic metrics and qualitative assessment to measure correctness, descriptiveness, and naturalness.
