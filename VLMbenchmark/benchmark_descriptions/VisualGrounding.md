# Visual Grounding Benchmark

The goal of **visual grounding** is to evaluate a vision-language model's ability to connect natural-language expressions with their corresponding regions in an image. Unlike **image captioning** or **VQA**, which mainly produce text outputs, visual grounding requires the model to **locate the visual referent of a phrase** by predicting spatial coordinates, bounding boxes, or regions.

Visual grounding is a broader concept that connects language and vision. For example, given the instruction *“find the person wearing a red jacket next to the bicycle”*, the model must understand the text, identify the correct person, and localize that instance in the image. It is a foundation for tasks such as referring expression comprehension, robotics, image editing, and human-computer interaction.

A visual grounding benchmark should evaluate:

* **Localization accuracy:** Does the predicted region correctly correspond to the described object or region?
* **Language-image alignment:** Does the model understand the relationship between the text description and the visual content?
* **Disambiguation ability:** Can the model distinguish between multiple similar objects in a scene?
* **Compositional reasoning:** Can the model ground complex descriptions involving attributes, relationships, and spatial constraints?

**Recommended benchmark datasets:**

* **RefCOCO** – A standard benchmark for referring expression comprehension, where models locate objects described by natural-language expressions.
* **RefCOCO+** – Similar to RefCOCO but contains more complex descriptions that avoid explicit location words, requiring stronger visual understanding.
* **RefCOCOg** – Uses longer, more natural descriptions generated through human annotations.
* **Flickr30k Entities** – Evaluates grounding of entity mentions in image captions to specific image regions.
* **Visual Genome** – Provides region descriptions and relationships for fine-grained visual-language grounding.

**Evaluation metrics:**

* **Acc@IoU (Accuracy at Intersection over Union threshold):** Measures whether the predicted bounding box overlaps sufficiently with the ground-truth region (commonly IoU ≥ 0.5).
* **mAP (mean Average Precision):** Used when evaluating multiple grounding predictions.
* **Recall@k:** Measures whether the correct region appears among the top-k predictions.
* **Pointing accuracy:** Evaluates whether the predicted point falls inside the correct object region.

**Example vision-language models capable of visual grounding:**

* Grounding DINO
* OWLv2
* GLIP
* Kosmos-2
* Florence-2
* Qwen3-VL
* GPT-4o

A visual grounding benchmark should assess whether a model can accurately map language to specific visual regions. Compared with **zero-shot object detection**, which typically detects objects from short category prompts (e.g., *“dog”* or *“car”*), visual grounding handles richer referring expressions (e.g., *“the small dog sitting under the table”*) and requires deeper understanding of language, context, and relationships between objects. Compared with **image segmentation**, grounding focuses on identifying *where* the described entity is rather than producing a precise pixel-level mask.
