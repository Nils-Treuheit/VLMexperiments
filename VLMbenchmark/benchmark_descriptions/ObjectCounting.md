# Object Counting Benchmark

The goal of **object counting** is to evaluate a vision-language model's ability to accurately count instances of objects specified by a natural-language prompt. Unlike **object detection**, which outputs object locations, object counting focuses on predicting the correct number of instances while often relying on detection or localization as an intermediate step. Modern VLMs can count objects by identifying each instance and, in some cases, tracking them across video frames.

An object counting benchmark should evaluate:

* **Counting accuracy:** Does the model predict the correct number of object instances?
* **Instance localization:** Can the model correctly identify individual objects before counting them?
* **Robustness:** Can the model count accurately in crowded scenes, with occlusions, or for small objects?
* **Generalization:** Can the model count arbitrary object categories specified by text prompts?

**Recommended benchmark datasets:**

* **FSC-147 (Few-Shot Counting 147)** – A standard benchmark for class-agnostic object counting.
* **CountBench** – Evaluates counting performance across diverse object categories and scenes.
* **CARPK** and **PUCPR+** – Benchmarks for counting objects in dense scenes such as parking lots.

**Evaluation metrics:**

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Counting Accuracy
* F1 score when instance localization is also evaluated

**Example vision-language models capable of object counting:**

* Molmo2
* Qwen3-VL
* GPT-4o

An object counting benchmark should assess both the accuracy of the predicted count and the model's ability to correctly identify individual object instances. Compared with zero-shot object detection, object counting emphasizes estimating the correct number of objects rather than producing precise object locations, although many modern vision-language models perform counting by first localizing each instance.
