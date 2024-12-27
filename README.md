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

### Additional Questions:
To adapt the large language model (LLM) to a classification task, the following steps were taken:

Data Preparation and Tokenization: First, we selected the IMDb dataset for sentiment analysis, as it provides a well-established, labeled text dataset suitable for binary sentiment classification. The text data was tokenized using the pre-trained DistilBERT tokenizer, ensuring it was properly formatted and padded/truncated to a fixed length for input compatibility with the model.

Model Selection and Fine-Tuning: I selected the DistilBERT model, a smaller, more efficient variant of BERT, to fine-tune it for the classification task. The model was initialized from the pre-trained distilbert-base-uncased checkpoint, and the output layer was adjusted to predict two classes (Positive and Negative).

Training and Evaluation: We set up a training pipeline using Hugging Face's Trainer class, specifying appropriate training arguments such as learning rate, batch size, and evaluation frequency. During training, the model was fine-tuned on the IMDb dataset to optimize its ability to classify sentiment from text.

### The primary differences in training the model as a classifier versus as a language model.
When training a model for classification as opposed to its original language modeling task, the primary differences include:

Output Layer Modification: Language models like BERT are trained to predict the next word in a sequence or masked tokens, whereas for classification tasks, the model’s output layer is adjusted to predict the class labels. This requires adding a classifier head (often a simple linear layer) on top of the pre-trained language model to map the learned representations to the desired output classes (positive/negative).

Training Objective: Language models are typically trained with unsupervised objectives like masked language modeling (MLM) or causal language modeling (CLM). For classification, the model is fine-tuned on a supervised task, where the objective is to minimize classification loss (e.g., cross-entropy loss) between the predicted labels and the true labels.

Data Format: In language modeling, the input is usually unstructured text, with the model learning to predict the next word in a sequence. For classification, the input is labeled data (e.g., movie reviews with sentiment labels), and the model learns to classify the text into predefined categories.

### Possible Improvements:

Ensemble Methods: Instead of relying solely on a single model, ensembling multiple models (e.g., fine-tuned versions of BERT, DistilBERT, and GPT-3.5) could provide a more robust prediction by leveraging their complementary strengths.

Advanced Fine-tuning Techniques: Techniques like learning rate scheduling or using a more advanced optimization algorithm (e.g., AdamW with weight decay) could improve model convergence and prevent overfitting.

Bigger and More Diverse Datasets: To improve generalization, incorporating more varied datasets and using domain-specific data for fine-tuning could make the model more adaptable to different types of text.


