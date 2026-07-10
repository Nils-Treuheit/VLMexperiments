### Human Intention Recognition Benchmark

The goal of **human intention recognition** is to evaluate a vision-language model's ability to infer a person's goals, plans, or likely next actions from visual observations. Unlike **object detection**, **visual grounding**, or **VQA**, which focus on identifying visible entities and answering explicit questions, intention recognition requires the model to interpret **implicit human behavior**, including body language, interactions, context, and environmental cues.

For example, given an image or video of a person reaching toward a door handle, the model should infer a possible intention such as *“the person is about to open the door.”* This task requires combining visual understanding with commonsense knowledge and often benefits from temporal information in videos.

A human intention recognition benchmark should evaluate:

* **Intent classification accuracy:** Can the model correctly identify the person's intended action or goal?
* **Context understanding:** Can the model use surrounding objects, environment, and social context to infer intent?
* **Future action prediction:** Can the model predict likely next actions based on current observations?
* **Human-object interaction understanding:** Can the model recognize how people interact with objects and other agents?
* **Uncertainty handling:** Can the model distinguish between observed actions and uncertain intentions?

**Recommended benchmark datasets:**

* **Intentonomy** – A benchmark for recognizing human intentions from images using hierarchical intent categories.
* **HICO-DET (Human-Object Interaction Detection)** – Evaluates recognition of human actions involving objects (e.g., *person riding bicycle*, *person holding cup*).
* **V-COCO** – Focuses on human-object interactions and action recognition.
* **EPIC-Kitchens** – A large-scale egocentric video dataset for understanding human actions and intentions during daily activities.
* **Ego4D** – Evaluates long-term first-person video understanding, including forecasting and activity reasoning.
* **Charades / Charades-Ego** – Benchmarks human activities and interactions in indoor environments.

**Evaluation metrics:**

* Accuracy and F1 score for intent classification
* Mean Average Precision (mAP) for multi-label action recognition
* Recall@k for future intention prediction
* Top-1 / Top-5 accuracy for action forecasting
* Temporal localization metrics for video-based intent recognition

**Example vision-language models capable of human intention recognition:**

* GPT-4o
* Qwen3-VL
* Gemini
* Video-LLaVA
* VideoChat
* InternVideo2

A human intention recognition benchmark should measure whether a model can move beyond recognizing **what is happening** to understanding **why it is happening and what is likely to happen next**. Compared with **VQA**, intention recognition usually involves implicit questions (*“What is this person trying to do?”*) rather than explicit queries. Compared with **video understanding**, it focuses specifically on interpreting human goals, behaviors, and future actions rather than general event comprehension.

**Typical benchmark task formulation:**

**Input:**

* Image or video clip
* Optional context or textual prompt

**Output:**

* Human intention label, action category, or natural-language explanation

**Example:**

* **Input:** Video of a person opening a refrigerator and reaching inside
* **Question:** "What is the person likely intending to do?"
* **Output:** "The person intends to take food or a drink from the refrigerator."

For VLM evaluation, this task is particularly valuable because it tests whether models can combine **visual perception + temporal reasoning + commonsense knowledge**, moving closer to human-level scene understanding.
