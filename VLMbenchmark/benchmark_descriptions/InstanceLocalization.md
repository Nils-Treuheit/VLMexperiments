### Instance Localization Benchmark

The goal of **instance localization** is to evaluate a vision-language model's ability to identify and localize objects in an image based on natural-language prompts. Unlike **image captioning**, **VQA**, or **visual reasoning**, which produce textual outputs, instance localization requires the model to output **spatial information**, such as bounding boxes, segmentation masks, or object coordinates. Modern vision-language models enable **zero-shot localization**, allowing them to detect or segment objects specified by text without task-specific training.

An instance localization benchmark should evaluate:

* **Localization accuracy:** Does the model correctly identify the location of the requested object?
* **Detection quality:** Are the predicted bounding boxes or segmentation masks accurate?
* **Open-vocabulary capability:** Can the model localize objects described by arbitrary text prompts, including unseen categories?
* **Generalization:** Can the model perform robust localization across diverse scenes and object types?

**Recommended benchmark datasets:**

* **COCO** – The standard benchmark for object detection and instance segmentation.
* **LVIS** – Evaluates long-tail and open-vocabulary object localization.
* **RefCOCO**, **RefCOCO+**, and **RefCOCOg** – Evaluate referring expression comprehension by localizing objects described in natural language.
* **ODinW (Objects in the Wild)** – Evaluates zero-shot and open-vocabulary object detection across diverse domains.

**Evaluation metrics:**

* Mean Average Precision (mAP) for object detection and segmentation
* Intersection over Union (IoU)
* Recall and Precision
* Grounding accuracy for referring expression localization

**Example vision-language models capable of instance localization:**

* Grounding DINO
* Florence-2
* Molmo2
* Qwen3-VL
* GPT-4o

An instance localization benchmark should assess a model's ability to accurately ground natural-language descriptions in images by predicting object locations or segmentation masks. Compared with other vision-language tasks, instance localization focuses on **precise spatial grounding** rather than generating text, enabling applications such as zero-shot object detection, referring expression comprehension, and image-guided detection.
