# Zero-Shot Image Classification

## Overview

The goal of **zero-shot image classification** is to evaluate a vision-language model's ability to classify an image into one of a set of **natural-language category descriptions** without task-specific training. Instead of predicting classes learned during supervised training, zero-shot classifiers leverage language understanding to recognize both familiar and previously unseen categories by matching images with text prompts.

Unlike traditional image classifiers, which require labeled training examples for every class, zero-shot image classification supports **open-vocabulary classification**, allowing models to identify arbitrary categories described in natural language.

### Zero-Shot Image Classification vs. Zero-Shot Object Detection

Although both tasks evaluate open-vocabulary vision-language understanding, they measure different capabilities.

| Zero-Shot Image Classification | Zero-Shot Object Detection |
|--------------------------------|----------------------------|
| Predicts one or more labels for an entire image. | Detects and localizes individual objects specified by text prompts. |
| Output is a class label (or ranked list of labels). | Output is one or more bounding boxes with associated labels and confidence scores. |
| Does not require object localization. | Requires accurate localization of each detected object. |
| Evaluates semantic recognition at the image level. | Evaluates both semantic recognition and spatial grounding. |
| Example: "This image contains a golden retriever." | Example: "Locate the golden retriever in this image." |

For example, given an image containing a dog sitting beside a bicycle:

- **Zero-shot image classification:** predicts *dog*, *golden retriever*, or *pet* as image-level labels.
- **Zero-shot object detection:** returns bounding boxes around the dog and bicycle when prompted with those object descriptions.

---

## Benchmark Objectives

A zero-shot image classification benchmark should evaluate:

- **Classification accuracy:** Does the model correctly identify the image category specified by the text prompt?
- **Open-vocabulary generalization:** Can the model correctly recognize categories that were not explicitly seen during supervised training?
- **Semantic understanding:** Can the model distinguish between visually similar categories using natural-language descriptions?
- **Confidence calibration:** Does the model assign reliable confidence scores that reflect prediction correctness?
- **Prompt robustness:** Is performance stable across different phrasings or synonyms describing the same category?

---

## Recommended Benchmark Datasets

### ImageNet

The standard benchmark for image classification. Frequently used for zero-shot evaluation using natural-language prompts corresponding to ImageNet classes.

### ImageNet-A

Evaluates robustness on naturally occurring, challenging examples that are difficult for standard classifiers.

### ImageNet-R

Measures generalization to artistic renditions such as paintings, sketches, cartoons, and sculptures.

### ImageNet-V2

Provides an independently collected test set for measuring reproducibility and generalization.

### CIFAR-10 and CIFAR-100

Useful for lightweight benchmarking and comparison across diverse categories.

### DomainNet

Evaluates cross-domain zero-shot recognition across photographs, sketches, clipart, paintings, and other visual domains.

### ObjectNet

Tests robustness to viewpoint changes, backgrounds, and object configurations not commonly represented in ImageNet.

### iNaturalist

Measures fine-grained zero-shot recognition across species with many rare categories.

---

## Evaluation Protocol

1. Prepare a fixed set of candidate category names.
2. Convert each category into one or more natural-language prompts.
3. Encode image and text representations using the vision-language model.
4. Compute image-text similarity scores.
5. Rank candidate labels according to similarity.
6. Compare predictions with ground-truth annotations.

Prompt ensembles (multiple textual templates per class) may be averaged to improve robustness.

### Example Prompt Templates

- "a photo of a {class}"
- "an image of a {class}"
- "this is a {class}"
- "a close-up photograph of a {class}"

---

## Evaluation Metrics

### Top-1 Accuracy

Percentage of images where the highest-scoring prediction matches the ground-truth class.

### Top-5 Accuracy

Percentage of images where the correct class appears among the five highest-scoring predictions.

### Precision

Fraction of predicted positive labels that are correct.

### Recall

Fraction of ground-truth labels correctly identified.

### F1 Score

Harmonic mean of precision and recall.

### Mean Per-Class Accuracy

Average classification accuracy across all categories, reducing bias toward frequent classes.

### Expected Calibration Error (ECE)

Measures how well predicted confidence scores correspond to actual classification accuracy.

### Negative Log-Likelihood (NLL)

Evaluates the quality of probabilistic predictions.

---

## Open-Vocabulary Evaluation

A comprehensive benchmark should assess performance on categories outside the model's supervised training distribution by evaluating:

- Novel object categories
- Fine-grained categories
- Rare classes
- Long-tail distributions
- Domain shifts
- Alternative textual descriptions
- Synonyms and paraphrases

This measures the model's ability to transfer semantic knowledge rather than memorize predefined label sets.

---

## Example Vision-Language Models for Zero-Shot Image Classification

- CLIP
- SigLIP
- SigLIP 2
- ALIGN
- EVA-CLIP
- MetaCLIP
- Florence-2
- Qwen3-VL
- GPT-4o

---

## Benchmark Reporting

Results should report:

- Top-1 Accuracy
- Top-5 Accuracy
- Per-class accuracy
- Precision
- Recall
- F1 Score
- Calibration metrics (ECE, NLL)
- Performance on seen vs. unseen categories
- Performance across different prompt templates
- Domain-specific performance (e.g., natural images, artwork, sketches)

---

## Summary

A zero-shot image classification benchmark evaluates a model's ability to assign semantic labels to entire images using natural-language category descriptions without additional training. In contrast, **zero-shot object detection** requires the model to both recognize and **localize** objects within an image by predicting bounding boxes corresponding to text prompts. While image classification focuses on **image-level semantic recognition**, object detection additionally evaluates **spatial grounding and localization**, making it a more complex task. Together, these benchmarks provide complementary assessments of a vision-language model's open-vocabulary understanding and generalization capabilities.
del's open-vocabulary understanding and generalization capabilities.