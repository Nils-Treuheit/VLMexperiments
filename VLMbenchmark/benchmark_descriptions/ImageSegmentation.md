### Image Segmentation Benchmark

The goal of **image segmentation** is to evaluate a vision-language model's ability to identify and precisely outline objects or regions in an image based on natural-language prompts. Unlike **object detection**, which predicts bounding boxes, segmentation requires the model to generate detailed pixel-level masks describing the exact shape and boundaries of the target object.

A segmentation benchmark should evaluate:

* **Mask accuracy:** Does the predicted segmentation mask correctly match the object's pixels?
* **Prompt grounding:** Can the model segment objects described by open-ended text prompts?
* **Boundary precision:** Can the model capture fine details such as object edges and complex shapes?
* **Generalization:** Can the model segment unseen object categories without task-specific training?

**Recommended benchmark datasets:**

* **COCO Instance Segmentation** – The standard benchmark for object-level segmentation.
* **LVIS** – Evaluates segmentation across a large number of categories with long-tail distributions.
* **RefCOCO**, **RefCOCO+**, and **RefCOCOg** – Evaluate segmentation based on natural-language referring expressions.
* **SA-1B (Segment Anything 1 Billion)** – A large-scale segmentation dataset used for training and evaluating general segmentation models.

**Evaluation metrics:**

* Intersection over Union (IoU)
* Mean IoU (mIoU)
* Average Precision (AP) for instance segmentation
* Boundary-based metrics for fine-grained mask quality

**Example vision-language models capable of image segmentation:**

* PaliGemma2
* Moondream2
* SAM 2 with vision-language prompting
* Qwen3-VL
* GPT-4o

An image segmentation benchmark should measure how effectively a model converts natural-language descriptions into precise pixel-level object masks. Compared with object detection, segmentation requires finer spatial understanding and produces richer visual outputs, enabling applications such as image editing, medical imaging, robotics, and detailed scene understanding.
