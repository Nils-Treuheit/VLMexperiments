### Video Understanding Benchmark

The goal of **video understanding** is to evaluate a vision-language model's ability to interpret and reason over visual content that evolves over time. Unlike **image understanding**, which operates on a single static image, video understanding requires the model to capture **temporal relationships**, including object motion, interactions, and events across multiple frames.

A video understanding benchmark should evaluate:

* **Temporal understanding:** Can the model correctly interpret actions and events across time?
* **Spatial-temporal reasoning:** Can it relate objects, movements, and interactions between frames?
* **Long-context understanding:** Can the model maintain context over long videos?
* **Temporal localization:** Can it identify when a particular event occurs within a video?

**Recommended benchmark datasets:**

* **VideoMME** – A benchmark for general video understanding across short, medium, and long videos.
* **LVBench** – Evaluates long-video understanding and reasoning.
* **TemporalBench** – Measures temporal reasoning and event localization.
* **MSR-VTT** – A standard benchmark for text-to-video retrieval.

**Evaluation metrics:**

* Task accuracy for video question answering and understanding
* Recall@k for text-to-video retrieval
* Temporal Intersection over Union (tIoU) and localization accuracy for temporal grounding
* F1 score for video pointing and tracking tasks

**Example vision-language models capable of video understanding:**

* Qwen3-VL
* Qwen3.5
* Molmo2
* SAGE
* GPT-4o

Video understanding encompasses several related tasks, including **video question answering**, **text-to-video retrieval**, and **temporal grounding**, where models identify the timestamps corresponding to a textual query. A video understanding benchmark should therefore assess both semantic understanding of video content and the ability to reason over temporal information. Compared with image-based vision-language tasks, video understanding introduces the additional challenge of modeling events and relationships that unfold over time.
