# Customer Support Text Classification: NLP Sequence Modeling

A comprehensive project implementing sentiment classification for customer support messages using traditional ML and deep learning approaches. This project explores the complete NLP pipeline from text preprocessing to transformer concepts.

## 📋 Project Overview

This project implements a complete NLP pipeline for multi-class sentiment classification (positive, neutral, negative) on customer support messages. We compare traditional vectorization methods with sequence-based deep learning models.

**Dataset:** 1,500 customer support messages from multiple channels (chat, email, phone, social, app)

## 🎯 Project Goals

1. **Dataset Understanding** - Load and analyze text data characteristics
2. **Text Preprocessing** - Clean and normalize text data
3. **Text Vectorization** - Convert text to numerical formats (BoW, TF-IDF, Sequences)
4. **Baseline Models** - Implement traditional ML classifiers
5. **Sequence Models** - Build LSTM for sequential processing
6. **Theoretical Understanding** - Explain RNN limitations and transformer advantages

## 📁 Project Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md                          # This file
├── notebook.ipynb                     # Main Jupyter notebook with all tasks
├── requirements.txt                   # Python dependencies
│
└── results/
    ├── 01_class_distribution.png              # Class balance analysis
    ├── 02_text_length_distribution.png        # Message length statistics
    ├── 03_channel_distribution.png            # Channel breakdown
    ├── 04_baseline_comparison.png             # LR vs RF accuracy
    ├── 05_baseline_confusion_matrices.png     # Error matrices for baselines
    ├── 06_lstm_training_history.png           # Learning curves
    ├── 07_lstm_confusion_matrix.png           # LSTM predictions
    ├── 08_all_models_comparison.png           # Complete model comparison
    ├── 09_attention_transformers_reflection.txt  # Detailed explanation
    ├── 10_sample_predictions.csv              # 20 test predictions
    ├── 11_evaluation_report.txt               # Comprehensive evaluation
    └── 12_complete_analysis_summary.png       # Dashboard summary
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Project

```bash
# Launch Jupyter notebook
jupyter notebook notebook.ipynb

# Or run all cells at once (requires nbconvert)
jupyter nbconvert --to notebook --execute notebook.ipynb
```

## 📊 Key Results

### Model Performance Comparison

| Model | Accuracy | Type | Training Time |
|-------|----------|------|---------------|
| Logistic Regression (TF-IDF) | 0.7867 | Traditional ML | 2 sec |
| Random Forest (BoW) | 0.7867 | Traditional ML | 5 sec |
| **Bidirectional LSTM** | **0.8400** | Deep Learning | 2 min |

The Bidirectional LSTM outperforms baseline models by **5.3%**, demonstrating the value of sequence modeling for text classification.

### Dataset Statistics

- **Total Records:** 1,500
- **Training Set:** 1,200 (80%)
- **Test Set:** 300 (20%)
- **Classes:** 3 (positive: 500, neutral: 450, negative: 550)
- **Average Message Length:** 12.8 words
- **Channels:** chat, email, phone, social, app

## 🔧 Tasks Explained

### Task 1: Dataset Understanding
- Loaded 1,500 customer messages
- Analyzed class distribution (balanced across three sentiments)
- Examined text statistics (length, channel distribution)
- Visualized data characteristics

### Task 2: Text Preprocessing
- **Lowercasing:** Normalize case variations
- **Special character removal:** Clean non-textual elements
- **Tokenization:** Break text into words
- **Stopword removal:** Remove common words
- **Lemmatization:** Reduce words to base forms

**Before:** "The refund process was fast and convenient. I appreciate the quick response."
**After:** "refund process fast convenient appreciate quick response"

### Task 3: Text Vectorization

#### Bag of Words (CountVectorizer)
- Shape: (1500, 1000)
- Counts word occurrences
- High sparsity: ~99.5%
- Fast but loses word order

#### TF-IDF Vectorizer
- Shape: (1500, 1000)
- Weights words by importance
- Reduces impact of frequent words
- Better for traditional ML models

#### Tokenizer-Based Sequences
- Shape: (1500, 100) - padded sequences
- Vocabulary size: 5,000 words
- Preserves word order
- Enables LSTM processing

**Why Vectorization?**
- ML models require numerical inputs
- Vectors capture semantic meaning
- Different methods suit different architectures
- Enables mathematical operations

### Task 4: Baseline Models

#### Logistic Regression (TF-IDF)
```
Accuracy: 0.7867
- Interpretable coefficients
- Fast training and prediction
- Works well with sparse features
```

#### Random Forest (Bag of Words)
```
Accuracy: 0.7867
- Handles non-linear patterns
- Feature importance ranking
- Ensemble of decision trees
```

### Task 5: Sequence Model (LSTM)

#### Architecture Overview
```
Input (padded sequences, 100 tokens)
    ↓
Embedding Layer (5000 vocab → 100-dim vectors)
    ↓
Bidirectional LSTM (64 units × 2 = 128 output)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units × 2 = 64 output)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (3 units, Softmax)
    ↓
Classification (Positive, Neutral, Negative)
```

#### Why LSTM?
- **Memory cells:** Long-term dependency learning
- **Gates:** Selective information flow
- **Bidirectional:** Context from both directions
- **Vanishing gradient:** Solved by additive connections

#### Performance
```
Test Accuracy: 0.8400
Test Loss: 0.4234
Training Epochs: 12 (early stopping)
```

### Task 6: Attention & Transformers Reflection

#### RNN Limitations
- **Vanishing Gradients:** Gradients decay exponentially (~0.97^100 ≈ 0.007)
- **Sequential Processing:** Cannot parallelize (slow)
- **Information Bottleneck:** Final state must encode entire sequence

#### LSTM Solutions
- **Memory Cells:** Additive connections preserve gradients
- **Gate Mechanisms:** Input, forget, output gates control flow
- **Long-range Dependencies:** Can learn relationships ~100+ tokens apart
- **Bidirectional:** Process context from both directions

**LSTM Cell Equations:**
```
i_t = σ(W_ii·x_t + W_hi·h_{t-1} + b_i)         # Input gate
f_t = σ(W_if·x_t + W_hf·h_{t-1} + b_f)         # Forget gate
o_t = σ(W_io·x_t + W_ho·h_{t-1} + b_o)         # Output gate
C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(W_ic·...)      # Cell state (KEY: additive)
h_t = o_t ⊙ tanh(C_t)                          # Hidden state
```

#### Attention Mechanism
- **Purpose:** Direct attention between distant positions
- **Formula:** Attention(Q,K,V) = softmax(QK^T/√d_k)V
- **Benefits:** 
  - Parallel processing
  - Interpretable (attention maps)
  - No information bottleneck
  - Better long-range dependencies

#### Why Transformers Are Revolutionary
1. **Parallelization:** Process entire sequence at once (10-100x faster)
2. **No Recurrence:** Self-attention replaces sequential processing
3. **Transfer Learning:** Pre-trained models (BERT, GPT) ready-to-use
4. **Generative AI Foundation:** Powers ChatGPT, Claude, etc.

**Transformer vs RNN/LSTM:**
```
Aspect              | RNN/LSTM          | Transformer
-------------------|-------------------|-----------
Parallelization     | Sequential (slow) | Parallel (fast)
Long-range deps     | Limited ~100      | Full sequence
Training time       | Hours/days        | Minutes/hours
Interpretability    | Gradient-based    | Attention maps
Transfer learning   | Limited           | Excellent
```

#### Modern NLP Timeline
1. **1990s-2000s:** RNNs (limited by vanishing gradients)
2. **2014-2016:** LSTMs & GRUs (solve gradients, still sequential)
3. **2017:** Transformers (parallel, attention-based)
4. **2018-2024:** Pre-trained Transformers (BERT, GPT-2/3, T5)
   - Foundation for Generative AI
   - Few-shot learning capabilities
   - State-of-the-art on nearly all NLP tasks

## 📈 Visualization Examples

### Model Comparison
The project generates comparison charts showing:
- Class distribution (class imbalance analysis)
- Text length statistics
- Channel usage patterns
- Model accuracy rankings
- Confusion matrices for error analysis
- Training curves (accuracy & loss over epochs)

### Key Insights
1. **Balanced Classes:** Similar distribution across sentiments
2. **Message Length:** Majority of messages 8-18 words (good for LSTM)
3. **Channel Distribution:** Chat and email are primary channels
4. **Deep Learning Advantage:** LSTM captures semantic patterns better

## 💡 Recommendations for Improvement

### 1. Pre-trained Embeddings (+3-5% accuracy)
```python
from tensorflow.keras.layers import Embedding
# Use Word2Vec, GloVe, or FastText pre-trained vectors
embedding_matrix = load_pretrained_embeddings()
```

### 2. Attention Layer (+2-3% accuracy)
```python
from tensorflow.keras.layers import Attention
# Add attention between LSTM output and input
attention = Attention()([lstm_output, lstm_output])
```

### 3. Transformer Models (+5-10% accuracy)
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Fine-tune BERT/DistilBERT on customer messages
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
```

### 4. Data Augmentation (+2-4% accuracy)
- Generate synthetic examples via back-translation
- Use synonym replacement
- Paraphrase minority classes

### 5. Ensemble Methods (+1-2% accuracy)
- Combine LSTM + Logistic Regression predictions
- Voting classifier with different architectures
- Stacking meta-learner

### 6. Class Weight Balancing
```python
class_weights = {0: 1.2, 1: 1.0, 2: 1.1}  # Weight minority classes
model.fit(..., class_weight=class_weights)
```

## 🔍 How to Interpret Results

### Classification Metrics
- **Accuracy:** Overall correctness (not reliable with imbalanced classes)
- **Precision:** Of positive predictions, how many are correct?
- **Recall:** Of actual positives, how many did we find?
- **F1-Score:** Harmonic mean of precision and recall

### Confusion Matrix
Shows where the model makes mistakes:
- **Diagonal:** Correct predictions
- **Off-diagonal:** Errors (false positives, false negatives)

### Sample Predictions CSV
Columns:
- `original_text`: Raw customer message
- `cleaned_text`: Preprocessed text
- `true_label`: Actual sentiment
- `predicted_label`: Model prediction
- `*_prob`: Confidence scores for each class

## 🎓 Learning Outcomes

After completing this project, you will understand:

1. ✓ How to load and analyze text datasets
2. ✓ Text preprocessing techniques and their importance
3. ✓ Multiple text vectorization approaches
4. ✓ Building and training traditional ML classifiers
5. ✓ LSTM architecture and bidirectional processing
6. ✓ Why RNNs have limitations (vanishing gradients)
7. ✓ How LSTMs solve the gradient problem
8. ✓ Attention mechanism and its advantages
9. ✓ Why transformers are revolutionary for NLP
10. ✓ Practical considerations for production deployment

## 🛠️ Technologies Used

- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML Models:** scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **NLP:** NLTK
- **Utilities:** Jupyter, IPython

## 📚 Further Resources

### Books
- "Deep Learning for NLP" by Delip Rao & Brian McMahan
- "Natural Language Processing in Action" by Lane et al.
- "Attention is All You Need" (Original Transformer Paper)

### Papers
- Hochreiter & Schmidhuber (1997): "LSTM Networks"
- Vaswani et al. (2017): "Attention is All You Need"
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"

### Courses
- Fast.ai NLP Course
- Stanford CS224N: NLP with Deep Learning
- DeepLearning.AI NLP Specialization

### Tools
- Hugging Face Transformers: `pip install transformers`
- TensorFlow Text: Advanced text preprocessing
- spaCy: Production NLP library

## 📝 License

This educational project is open-source. Feel free to modify and use for learning purposes.

## 🤝 Contributing

To improve this project:
1. Try alternative architectures (GRU, Attention, Transformer)
2. Test different preprocessing techniques
3. Experiment with hyperparameter tuning
4. Implement additional evaluation metrics
5. Add cross-validation for robustness

## 📞 Questions & Support

For questions about:
- **Text preprocessing:** See Task 2 in notebook
- **Vectorization:** See Task 3 explanation
- **Model architecture:** See LSTM section with diagrams
- **Theoretical concepts:** See Task 6 reflection document

---

**Project Status:** ✅ Complete  
**Last Updated:** 2026  
**Python Version:** 3.8+  
**TensorFlow Version:** 2.10+
