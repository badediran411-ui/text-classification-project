# News Text Classification

A multi-class text classification project that categorizes news articles into **World**, **Sports**, **Business**, and **Sci/Tech** using the AG News dataset. Three models were built and compared: Logistic Regression, CNN, and DistilBERT.

## Dataset

- **Source:** AG News (`fancyzhx/ag_news` on Hugging Face)
- **Training set:** 120,000 articles (30,000 per class)
- **Test set:** 7,600 articles (1,900 per class)
- **Classes:** 0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech

## Models

| Model | Approach |
|---|---|
| Logistic Regression | TF-IDF vectorization (10,000 features) |
| CNN | Embedding + Conv1D + GlobalMaxPooling |
| DistilBERT | Fine-tuned `distilbert-base-uncased`, trained on a balanced 30,000-row subset |

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 91.25% |
| CNN | 91.14% |
| DistilBERT | 93.30% |

![Model Comparison](plots/model_comparison.png)

DistilBERT performed best, benefiting from contextual language understanding that TF-IDF and basic embeddings don't capture.

## Project Structure

- lr_model.pkl              # Trained Logistic Regression model
- tfidf_vectorizer.pkl      # TF-IDF vectorizer
- cnn_model.keras           # Trained CNN model
- tokenizer.pkl             # Keras tokenizer for CNN
- distilbert_model/         # Trained DistilBERT model + tokenizer
- plots/                    # All generated plots (confusion matrices, ROC curves, training curves)
- REPORT.md                 # Full project report

## Requirements

- datasets
- scikit-learn
- tensorflow
- transformers
- accelerate
- matplotlib, seaborn

## Usage

Load any saved model with its corresponding preprocessing tool (vectorizer/tokenizer) to run predictions on new text. See REPORT.md for full details, including confusion matrices and per-class performance.
