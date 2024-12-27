# Open-LLM2Classifer

# Transforming DistilBERT into a Sentiment Classifier for IMDb Reviews

## Table of Contents
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
- Replaced the original output layer with a linear classification head for binary sentiment classification.

### Implementing Categorical Cross-Entropy Loss
- Used Hugging Face's `Trainer` class, which incorporates cross-entropy loss.

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


Performance Metrics Across Epochs
Epoch	Training Loss	Validation Loss	Accuracy	Precision	Recall	F1-Score
1	0.2468	0.8994	0.8988	0.8992	0.8990	0.8990
2	0.2876	0.4233	0.8678	0.8035	0.9723	0.8799
3	0.2876	0.3455	0.9134	0.9089	0.9181	0.9135
Model Saving
python
Copy code
model.save_pretrained("./distilbert-imdb-classifier")
tokenizer.save_pretrained("./distilbert-imdb-classifier")
Evaluation Metrics and Results
Metric	Fine-Tuned DistilBERT	GPT-3.5 Prompt-Based
Accuracy	91.34%	91.8%
Precision	0.9089	0.9622
Recall	0.9181	0.8605
F1-Score	0.9135	0.9085
Comparison with Prompt-Based Approach
Aspect	Fine-Tuned DistilBERT	GPT-3.5 Prompt-Based
Training	Fine-tuning required	None (prompt-only)
Inference Time	Faster (local execution)	Slower (API calls)
Cost	Free (after training)	Paid API usage
Flexibility	Sentiment-specific	High (adaptable)
Challenges and Future Work
Challenges Faced
API Rate Limits: Introduced delays to comply with GPT-3.5 API policies.
Overfitting: Weight decay employed to mitigate overfitting.
Resource Constraints: Limited GPU resources impacted training speed.
Future Improvements
Data Augmentation: Enrich dataset with synonym replacement or back-translation.
Hyperparameter Tuning: Explore broader hyperparameter ranges.
Model Ensemble: Combine multiple models for improved accuracy.
Advanced Prompt Engineering: Refine GPT-3.5 prompts.
Conclusion
This project successfully transformed DistilBERT into a sentiment classifier for IMDb reviews, achieving high accuracy and balanced performance metrics. The comparative analysis with GPT-3.5 demonstrated the advantages of fine-tuned models for domain-specific tasks. Future efforts will focus on addressing challenges and exploring enhancements to improve model performance.

Appendices
Appendix A: Evaluation Metrics Computation
python
Copy code
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    logits, labels = pred
    preds = torch.argmax(torch.tensor(logits), axis=1).cpu().numpy()
    labels = labels.cpu().numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
Appendix B: Prompt Template for GPT-3.5
python
Copy code
def create_prompt(review_text):
    return f"""
    The following text is a movie review. Classify it as either "Positive" or "Negative" based on the sentiment.
    Review: "{review_text}"
    Sentiment:
    """
