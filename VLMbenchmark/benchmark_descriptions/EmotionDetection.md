# Emotion Detection Benchmark

The goal of **emotion detection** is to evaluate a vision-language model's ability to recognize and interpret human emotional states from visual information, such as facial expressions, body language, gestures, and situational context. Unlike **human intention recognition**, which focuses on understanding a person's goals or future actions, emotion detection focuses on identifying **affective states** (e.g., happiness, sadness, anger, surprise, fear) expressed by individuals.

Emotion detection is a challenging multimodal task because emotions are often subtle, context-dependent, and culturally influenced. A model may need to combine facial cues, posture, interactions with other people, and surrounding context rather than relying only on a person's appearance.

A benchmark for emotion detection should evaluate:

* **Emotion recognition accuracy:** Can the model correctly identify the expressed emotion?
* **Fine-grained understanding:** Can the model distinguish between similar emotional states (e.g., frustration vs. anger, surprise vs. fear)?
* **Context awareness:** Can the model interpret emotions using surrounding scene information and social interactions?
* **Multimodal reasoning:** Can the model combine facial expressions, body language, and situational cues?
* **Emotion explanation:** Can the model provide a grounded explanation for its prediction when required?

**Recommended benchmark datasets:**

* **AffectNet** – A large-scale facial expression dataset with categorical emotion labels and continuous valence/arousal annotations.
* **FER2013** – A widely used facial emotion recognition benchmark with seven basic emotion categories.
* **RAF-DB (Real-world Affective Faces Database)** – Contains real-world facial expressions with basic and compound emotion labels.
* **Emotic** – Focuses on recognizing emotions from people in context, including body posture and surrounding scenes.
* **HAPPEI / HUMAINE / MELD** – Useful for multimodal emotion recognition involving speech, text, and video.
* **MAFW (Multi-modal Affective Facial Expression Database)** – Evaluates more complex facial expressions and affective states.

**Evaluation metrics:**

* Accuracy
* Macro F1 score (important for imbalanced emotion categories)
* Precision and Recall
* Concordance Correlation Coefficient (CCC) for continuous valence/arousal prediction
* Mean Squared Error (MSE) for regression-based emotion estimation

**Example vision-language models capable of emotion detection:**

* GPT-4o
* Qwen3-VL
* Gemini
* LLaVA
* Video-LLaVA
* InternVideo2

A human emotion detection benchmark should assess whether a model can interpret emotional signals while avoiding overconfident assumptions from limited visual evidence. Compared with **VQA**, emotion detection typically asks the model to infer an internal state rather than answer an explicit factual question. Compared with **human intention recognition**, emotion detection focuses on *how someone feels*, whereas intention recognition focuses on *what someone is trying to do*. Compared with **facial expression recognition**, modern VLM-based emotion detection can incorporate broader context beyond facial features alone.

**Typical benchmark task formulation:**

**Input:**

* Image or video containing one or more people
* Optional contextual prompt

**Output:**

* Emotion category, affective dimensions, or natural-language interpretation

**Examples:**

**Image input:**
A person smiling while holding a graduation certificate.

**Output:**

* Emotion: *Happiness / Pride*
* Explanation: *The person's facial expression and context suggest a positive emotional state associated with achievement.*

**Video input:**
A person repeatedly looking at a broken device and sighing.

**Output:**

* Emotion: *Frustration*
* Explanation: *The person's repeated attempts and body language suggest difficulty and dissatisfaction.*

For VLM evaluation, emotion detection tests whether models can move beyond recognizing visible objects and actions toward understanding **human-centered, socially meaningful visual cues**.
