# Open-LLM2Classifier

## Transforming DistilBERT into a Sentiment Classifier for IMDb Reviews

### Table of Contents
1. [Overview](#overview)
2. [Dataset Documentation](#dataset-documentation)
3. [Model Selection and Preprocessing](#model-selection-and-preprocessing)
4. [Transforming DistilBERT into a Classifier](#transforming-distilbert-into-a-classifier)
5. [Training Process](#training-process)
6. [Evaluation Metrics and Results](#evaluation-metrics-and-results)
7. [Comparison with Prompt-Based Approach](#comparison-with-prompt-based-approach)
8. [Challenges and Future Work](#challenges-and-future-work)
9. [Conclusion](#conclusion)
10. [Appendices](#appendices)

---

## Overview

### Objective
The main objective of this project is to adapt an open-source Language Model (LLM), specifically DistilBERT, into a classifier tailored for sentiment analysis on the IMDb dataset. This exercise mirrors industry practices of repurposing generative models for classification tasks under constraints like limited data and restricted resources.

### Approach
- **Dataset Selection:** IMDb dataset chosen for relevance to sentiment analysis and manageable size.
- **Model Selection:** DistilBERT utilized for its efficiency and pre-trained capabilities.
- **Model Transformation:** Added a binary classification head to DistilBERT.
- **Training:** Fine-tuned on a subset of IMDb data using optimal hyperparameters.
- **Evaluation:** Performance compared to a prompt-based approach with GPT-3.5.
- **Documentation:** Comprehensive coverage of the process, challenges, and improvements.

---

## Dataset Documentation

### Selected Dataset: IMDb Movie Reviews
- **Source:** IMDb Dataset
- **Type:** Binary Classification (Positive/Negative)
- **Number of Samples:**
  - Training: 25,000
  - Testing: 25,000

### Preprocessing
- **Sampling:** Reduced to 5,000 samples each for training and testing.
- **Shuffling:** Used a fixed seed (40) for reproducibility.

### Rationale for Selection
1. **Relevance:** IMDb reviews provide clear binary sentiment labels.
2. **Size:** 50,000 samples allow manageable sampling.
3. **Benchmarking:** Widely used in NLP research for sentiment analysis.

---

## Model Selection and Preprocessing

### Model Selection: DistilBERT
#### Why DistilBERT?
- **Efficiency:** Lighter and faster than BERT with comparable performance.
- **Resource-Friendly:** Optimized for environments with limited resources.
- **Pre-trained:** Transfer learning reduces the need for extensive data.

### Preprocessing Steps
1. **Tokenization:** 
   - Used `DistilBertTokenizer` from Hugging Face.
   - Configured to truncate and pad to 512 tokens.
2. **Dataset Mapping:** 
   - Tokenized text and moved tensors to the appropriate device.
3. **Formatting:** 
   - Renamed label column to align with model input requirements.
   - Set dataset format to PyTorch tensors.

---

## Transforming DistilBERT into a Classifier

### Modifying the Output Layer
To transform DistilBERT into a sentiment classifier, the following steps were taken:
- **Replaced the original output layer** with a linear classification head for binary sentiment classification (Positive/Negative).
- **Implemented Categorical Cross-Entropy Loss** using Hugging Face's `Trainer` class, which automatically integrates loss calculation.

### Updating the Training Procedure
- **Hyperparameters:**
  - Learning Rate: `5e-5`
  - Batch Size: `16`
  - Epochs: `3`
  - Weight Decay: `0.01`
  - Mixed Precision: `fp16=True`
- **Trainer Configuration:** 
  - Integrated model, training arguments, tokenized datasets, and custom metrics.

---

## Training Process

### Training Setup
- **Environment:** Google Colab with GPU support.
- **Libraries:** PyTorch, Transformers, Datasets, Scikit-learn.

### Training Execution

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)
trainer.train()

### Performance Metrics Across Epochs
| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall | F1-Score |
|-------|---------------|-----------------|----------|-----------|--------|----------|
| 1     | 0.2468        | 0.8994          | 0.8988   | 0.8992    | 0.8990 | 0.8990   |
| 2     | 0.2876        | 0.4233          | 0.8678   | 0.8035    | 0.9723 | 0.8799   |
| 3     | 0.2876        | 0.3455          | 0.9134   | 0.9089    | 0.9181 | 0.9135   |

### Observations
- **Performance Improvement:** Significant increase in accuracy and F1-score from Epoch 1 to Epoch 3.
- **Validation Loss:** Decreased steadily, indicating effective learning throughout the epochs.
- **Metrics Stability:** Precision, recall, and F1-score remained consistent, showcasing balanced model performance.

---

## Evaluation Metrics and Results

### Metrics Used
- **Accuracy:** Proportion of correct predictions out of total predictions.
- **Precision:** Ratio of true positive predictions to the total positive predictions.
- **Recall:** Ratio of true positive predictions to all actual positives.
- **F1-Score:** Harmonic mean of precision and recall, providing a balance between the two.

### Evaluation Results

#### Fine-Tuned DistilBERT:
- **Accuracy:** 91.34%
- **Precision:** 0.9089
- **Recall:** 0.9181
- **F1-Score:** 0.9135

#### GPT-3.5 Prompt-Based Approach:
- **Accuracy:** 91.8%
- **Precision:** 0.9622
- **Recall:** 0.8605
- **F1-Score:** 0.9085

---

## Comparison with Prompt-Based Approach

### Prompt-Based Classification Using GPT-3.5
**Methodology:**
- Employed OpenAI's GPT-3.5 (gpt-3.5-turbo) to classify IMDb reviews by crafting specific prompts.

**Example Prompt:**
The following text is a movie review. Classify it as either "Positive" or "Negative" based on the sentiment. Review: "I absolutely loved this movie! The performances were stellar and the story was gripping." Sentiment: Positive


- Iterated over the evaluation dataset, sending each review to the GPT model and recording its prediction.
- Incorporated a delay (`time.sleep(1)`) to adhere to API rate limits.

### Evaluation Metrics:

| Metric      | Fine-Tuned DistilBERT | GPT-3.5 Prompt-Based |
|-------------|------------------------|-----------------------|
| **Accuracy**| 91.34%                | 91.8%                |
| **Precision**| 0.9089               | 0.9622               |
| **Recall**  | 0.9181                | 0.8605               |
| **F1-Score**| 0.9135                | 0.9085               |

### Performance Comparison

| **Aspect**         | **Fine-Tuned DistilBERT**  | **GPT-3.5 Prompt-Based** |
|---------------------|---------------------------|--------------------------|
| Accuracy            | Slightly lower at 91.34% | Slightly higher at 91.8% |
| Precision           | Lower at 0.9089          | Higher at 0.9622         |
| Recall              | Higher at 0.9181         | Lower at 0.8605          |
| F1-Score            | Comparable at 0.9135     | Comparable at 0.9085     |

### Conclusion:
- The fine-tuned DistilBERT model demonstrates **higher recall**, making it better for tasks where capturing all relevant instances is critical.
- The GPT-3.5 prompt-based approach excels in **precision**, which is useful when false positives need to be minimized.
- Both approaches provide comparable F1-scores, showcasing balanced performance.

---

## Challenges and Future Work

### Challenges:
- **Imbalanced Dataset:** One challenge was the imbalance between positive and negative reviews in the IMDb dataset. This could affect model performance, especially in terms of precision and recall. We used a variety of metrics to ensure balanced evaluation, but further techniques like oversampling or class weighting could improve model fairness.
- **Training Efficiency:** Fine-tuning a large model like DistilBERT, even though efficient, required significant computational resources. Future work could explore quantization or pruning methods to make the model even more resource-efficient.

### Future Work:
- **Fine-Grained Sentiment Analysis:** Extend the model to multi-class sentiment classification, capturing a wider range of sentiment nuances.
- **Additional Datasets:** Test the model on other sentiment analysis datasets to improve generalization.


- Iterated over the evaluation dataset, sending each review to the GPT model and recording its prediction.
- Incorporated a delay (`time.sleep(1)`) to adhere to API rate limits.

### Evaluation Metrics:

| Metric      | Fine-Tuned DistilBERT | GPT-3.5 Prompt-Based |
|-------------|------------------------|-----------------------|
| **Accuracy**| 91.34%                | 91.8%                |
| **Precision**| 0.9089               | 0.9622               |
| **Recall**  | 0.9181                | 0.8605               |
| **F1-Score**| 0.9135                | 0.9085               |

### Performance Comparison

| **Aspect**         | **Fine-Tuned DistilBERT**  | **GPT-3.5 Prompt-Based** |
|---------------------|---------------------------|--------------------------|
| Accuracy            | Slightly lower at 91.34% | Slightly higher at 91.8% |
| Precision           | Lower at 0.9089          | Higher at 0.9622         |
| Recall              | Higher at 0.9181         | Lower at 0.8605          |
| F1-Score            | Comparable at 0.9135     | Comparable at 0.9085     |

### Conclusion:
- The fine-tuned DistilBERT model demonstrates **higher recall**, making it better for tasks where capturing all relevant instances is critical.
- The GPT-3.5 prompt-based approach excels in **precision**, which is useful when false positives need to be minimized.
- Both approaches provide comparable F1-scores, showcasing balanced performance.

---

## Challenges and Future Work

### Challenges:
- **Imbalanced Dataset:** One challenge was the imbalance between positive and negative reviews in the IMDb dataset. This could affect model performance, especially in terms of precision and recall. We used a variety of metrics to ensure balanced evaluation, but further techniques like oversampling or class weighting could improve model fairness.
- **Training Efficiency:** Fine-tuning a large model like DistilBERT, even though efficient, required significant computational resources. Future work could explore quantization or pruning methods to make the model even more resource-efficient.

### Future Work:
- **Fine-Grained Sentiment Analysis:** Extend the model to multi-class sentiment classification, capturing a wider range of sentiment nuances.
- **Additional Datasets:** Test the model on other sentiment analysis datasets to improve generalization.




