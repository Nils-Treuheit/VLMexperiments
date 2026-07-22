# Zero-Shot Object Detection Benchmark

The goal of **zero-shot object detection** is to evaluate a vision-language model's ability to detect and localize objects specified by **natural-language prompts** without task-specific training. Unlike traditional object detectors, which are trained on a fixed set of object classes, zero-shot detectors support **open-vocabulary detection**, enabling them to recognize previously unseen object categories through text descriptions.

A zero-shot object detection benchmark should evaluate:

* **Detection accuracy:** Does the model correctly detect the objects specified in the text prompt?
* **Localization quality:** Are the predicted bounding boxes accurate?
* **Open-vocabulary generalization:** Can the model detect novel object categories not seen during training?
* **Confidence calibration:** Does the model assign reliable confidence scores that balance precision and recall?

**Recommended benchmark datasets:**

* **LVIS** – Evaluates open-vocabulary detection on a large set of object categories.
* **ODinW (Objects in the Wild)** – Measures zero-shot object detection across diverse domains.
* **COCO** – Commonly used to evaluate object detection and zero-shot transfer.
* **RefCOCO**, **RefCOCO+**, and **RefCOCOg** – Evaluate localization from natural-language referring expressions.

**Evaluation metrics:**

* Mean Average Precision (mAP)
* Intersection over Union (IoU)
* Precision and Recall
* Average Recall (AR)

**Example vision-language models capable of zero-shot object detection:**

* Grounding DINO
* OWLv2
* Florence-2
* Qwen3-VL
* GPT-4o

A zero-shot object detection benchmark should measure how accurately a model grounds natural-language object descriptions to image regions without requiring additional training. Compared with traditional object detection, zero-shot detection supports arbitrary text prompts and open-vocabulary recognition, making it well suited for applications such as automatic dataset annotation, image search, and flexible object localization.
