# Foundations of Generative AI (ICT4403) - Assignment 2
## Comparative Autoregressive Text Generation System: Simple RNN, Bidirectional RNN, LSTM, and GRU

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Problem Statement and Objectives



In this project, we implement and empirically compare four core recurrent neural network architectures for next-word prediction:
1. **Simple RNN (Elman RNN)**
2. **Bidirectional RNN (BiRNN)**
3. **Long Short-Term Memory (LSTM)**
4. **Gated Recurrent Unit (GRU)**

All four architectures are trained under strictly controlled, identical experimental conditions on a literary corpus (*The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle) to predict the next word given a preceding context window. We evaluate:
- **Quantitative Metrics:** Cross-Entropy Loss, Perplexity ($\text{PPL} = \exp(\mathcal{L})$), Top-1 Accuracy, Top-5 Accuracy, parameter counts, and training runtimes.
- **Qualitative Metrics:** Grammatical coherence, syntactic structure, repetitive looping, and n-gram lexical diversity ($\text{Distinct-1}$ and $\text{Distinct-2}$).
- **Sampling-Temperature Analysis:** Generation behaviour across three sampling temperatures ($T = 0.3, 0.7, 1.2$) across identical seed sequences.

---

## 2. Dataset Description and Source Link

- **Corpus Title:** *The Adventures of Sherlock Holmes* (Sir Arthur Conan Doyle)
- **Source Link:** [Project Gutenberg eBook #1661](https://www.gutenberg.org/ebooks/1661) (Direct raw URL: `https://www.gutenberg.org/cache/epub/1661/pg1661.txt`)
- **Corpus Characteristics:**
  - **Raw Text Size:** ~594 KB (593,911 characters)
  - **Total Tokens:** 127,243 tokens
  - **Vocabulary Size:** 3,106 unique words (frequency threshold $\ge 3$, rare words mapped to `<unk>`)
  - **Special Tokens:** `<pad>` (index 0), `<unk>` (index 1), `<eos>` (index 2)
  - **Sliding Window:** Sequence length $L = 20$, stride $= 5$
  - **Dataset Split:** 25,445 sequence pairs (80% Training: 20,356 pairs; 20% Validation: 5,089 pairs)

---

## 3. Models Implemented

All four models share identical configurations:
- **Embedding Dimension ($D_{emb}$):** 128
- **Hidden Dimension ($D_{hid}$):** 128
- **Number of Recurrent Layers:** 1
- **Dropout Rate:** 0.1
- **Optimizer:** Adam ($\text{lr} = 0.001$, gradient clipping $\text{max\_norm} = 1.0$)
- **Loss Function:** Cross-Entropy Loss with padding ignored ($\text{ignore\_index} = 0$)

```
+------------------------------------------------------------------------------------+
| Architecture Summary                                                               |
+------------------+---------------------+-------------------+-----------------------+
| Model            | Hidden Update Gate  | Memory Cell       | Trainable Parameters  |
+------------------+---------------------+-------------------+-----------------------+
| Simple RNN       | Tanh                | None              | 831,266               |
| BiRNN            | Tanh (Fwd + Bwd)    | None              | 1,261,858             |
| LSTM             | Input, Forget, Out  | Cell State c_t    | 930,338               |
| GRU              | Reset, Update       | Candidate h~_t    | 897,314               |
+------------------+---------------------+-------------------+-----------------------+
```

### 3.1 Simple RNN (Elman RNN)
Updates hidden state using current token embedding and previous state:
$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

### 3.2 Bidirectional RNN (BiRNN)
Encodes prefix sequence in both forward ($\vec{h}_t$) and reverse ($\overleftarrow{h}_t$) directions:
$$h_t = [\vec{h}_t \,;\, \overleftarrow{h}_t]$$
The final concatenated hidden state encapsulates bidirectional context over the prefix to project to vocabulary logits.

### 3.3 Long Short-Term Memory (LSTM)
Maintains internal cell state $c_t$ governed by input gate $i_t$, forget gate $f_t$, and output gate $o_t$:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c), \quad h_t = o_t \odot \tanh(c_t)$$

### 3.4 Gated Recurrent Unit (GRU)
Employs update gate $z_t$ and reset gate $r_t$ without an isolated cell state:
$$z_t = \sigma(W_z [h_{t-1}, x_t]), \quad r_t = \sigma(W_r [h_{t-1}, x_t])$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t]), \quad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

---

## 4. Execution Steps and Required Dependencies

### 4.1 Prerequisites & Installation
Ensure Python 3.10+ is installed. Install all dependencies via:
```bash
pip install -r requirements.txt
```

Required packages:
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.8.0`
- `python-docx >= 1.1.0`
- `tqdm >= 4.66.0`

### 4.2 Running the Full Pipeline
To execute the complete end-to-end dataset download, preprocessing, training, evaluation, plot generation, and temperature analysis:
```bash
python main.py --epochs 10 --batch_size 128 --seq_len 20 --stride 5
```

### 4.3 Running the Jupyter Notebook
Open the interactive Jupyter Notebook with all outputs and figures:
```bash
jupyter notebook Autoregressive_Text_Generation.ipynb
```

### 4.4 Generating the Submission Document
Generate the Step 3 Word document submission template:
```bash
python create_submission_docx.py
```

---

## 5. Results Obtained

The models were evaluated over 10 epochs. Below is the quantitative performance summary:

| Model Architecture | Trainable Parameters | Best Val Loss | Best Val Perplexity (PPL) | Top-1 Accuracy | Top-5 Accuracy | Total Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Simple RNN** | 831,266 | 4.9447 | 140.43 | 18.12% | 38.44% | **52.6s** |
| **Bidirectional RNN** | 1,261,858 | 4.9730 | 144.46 | 17.80% | 38.93% | 88.4s |
| **LSTM** | 930,338 | 4.9494 | 141.09 | 18.22% | **40.17%** | 82.4s |
| **GRU** | **897,314** | **4.8902** | **132.98** | **18.26%** | 39.69% | 90.2s |

### Evaluation Plots
The generated publication-quality plots are stored in `plots/`:
1. `plots/loss_curves.png`: Training & validation loss trajectories across 10 epochs.
2. `plots/perplexity_and_accuracy.png`: Comparative validation perplexity and Top-1 / Top-5 next-word accuracy.
3. `plots/training_time_and_params.png`: Computational efficiency, parameter scaling, and training duration.
4. `plots/diversity_vs_temperature.png`: Lexical richness ($\text{Distinct-1}$ & $\text{Distinct-2}$) versus sampling temperature.

---

## 6. Model Comparison and Evaluation

1. **Loss & Perplexity:**
   - **GRU** achieved the lowest validation loss (**4.8902**) and lowest validation perplexity (**132.98**), demonstrating the highest probabilistic confidence on unseen text.
   - **LSTM** reached a strong validation perplexity of **141.09** and the highest Top-5 prediction accuracy (**40.17%**).
   - **Simple RNN** achieved a competitive perplexity (**140.43**) during early epochs but exhibited signs of gradient instability and slight overfitting after epoch 5 as training loss dropped to 3.43 while validation loss stagnated around 4.94.
   - **BiRNN** had the largest parameter footprint (1.26M parameters) due to dual forward and backward projection matrices. While expressive, the lack of gating mechanisms caused it to plateau at a higher perplexity (144.46).

2. **Computational Efficiency:**
   - **Simple RNN** was the fastest per epoch (~5.2s), but lacked gating capabilities.
   - **GRU** achieved superior perplexity with fewer parameters (897K) than LSTM (930K), making it the most parameter-efficient model.

---

## 7. Sampling-Temperature Analysis

Autoregressive decoding uses temperature scaling to reshape the next-token probability distribution:
$$P(w_{t+1} = i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

We evaluated generation across three temperatures on the seed:  
`"the detective looked carefully at the"`

### Sample Generations:
```
[T = 0.3 - Conservative / Low Temperature]
- Simple_RNN : "... door. "" i have been a very peculiar of the matter. "" i have been a very in a very serious"
- BiRNN      : "... matter. "" i had been to the whole of the door, and the whole, and i had not know that the"
- LSTM       : "... room. "" i am not see you. "" i have been no doubt, and i have been no doubt,"
- GRU        : "... matter. "" i am not have been to be the matter. "" i am that you have been not be to"

[T = 0.7 - Balanced / Optimal]
- Simple_RNN : "... case man. "it is possible do to each?" " the conviction of the man to me, and? ' here"
- BiRNN      : "... offices. ' "the neighbourhood, but i shall be a little person of the earth of us, it." i have"
- LSTM       : "... opening, and i am wished to the other side of the wife to the trees. one ' s of the window. ""
- GRU        : "... whether was happened in the cab manner. mrs. the case there is a little hand of front from the floor of his overcoat"

[T = 1.2 - Explorative / High Temperature]
- Simple_RNN : "... shutter station finger under the afternoon have is possible angel by the been only gone ever which at him..."
- BiRNN      : "... thumb of writing, holding east for i locked tell to speak example return vacancy into the pockets ran..."
- LSTM       : "... idea. ' hum put we - knee change i will my own dear that found, you many now present do off..."
- GRU        : "... daughter days eventually minute their or vacancy work piece. followed stalls among my ask, fuller so..."
```

### Influence of Temperature on Metrics:
- **Low Temperature ($T = 0.3$):**
  - **Coherence:** Grammatically tight at the local phrase level.
  - **Repetition:** Pronounced repetitive looping (e.g., *"i have been no doubt, and i have been no doubt"*).
  - **Diversity:** Low Distinct-1 ratio (~0.57 - 0.63).
- **Balanced Temperature ($T = 0.7$):**
  - **Coherence:** Strongest overall semantic and syntactic flow, reflecting the Victorian narrative cadence of Sherlock Holmes.
  - **Repetition:** Zero consecutive repetitive cycles.
  - **Diversity:** Well-balanced lexical richness (Distinct-1: ~0.85 - 0.90, Distinct-2: up to 1.0).
- **High Temperature ($T = 1.2$):**
  - **Coherence:** Syntactic degradation; words become semantically disconnected.
  - **Repetition:** Non-existent repetition.
  - **Diversity:** Maximal Distinct-1 and Distinct-2 ratios (0.96 - 1.00).

---

## 8. Conclusion Identifying the Best-Performing Model

Based on rigorous quantitative metrics and qualitative generation evaluation:

1. **Winning Architecture:** **Gated Recurrent Unit (GRU)** is selected as the best-performing model.
   - **Lowest Perplexity:** Achieved a validation perplexity of **132.98** (vs. 140.43 for RNN, 144.46 for BiRNN, and 141.09 for LSTM).
   - **Superior Accuracy:** Attained the highest Top-1 accuracy (**18.26%**) and robust Top-5 accuracy (**39.69%**).
   - **Parameter Efficiency:** GRU requires 33,024 fewer parameters than LSTM and 364,544 fewer than BiRNN while achieving lower loss.
   - **Memory Dynamics:** GRU's coupled reset and update gates effectively retain multi-step dependencies without the cell-state memory overhead of LSTM.
2. **Recommended Sampling Setting:**
   - **$T = 0.7$** provides the ideal operational temperature for autoregressive text generation on this dataset.

---

## 9. Repository Structure

```
ICT4403_Assignment2_Text_Generation/
│
├── Autoregressive_Text_Generation.ipynb    # Complete executed Jupyter Notebook
├── README.md                              # Comprehensive assignment documentation
├── requirements.txt                       # Project dependencies
├── main.py                                # Master CLI training & evaluation script
├── create_submission_docx.py              # Generator for Step 3 Word document
├── YourName_RegNum_FGAI_Assignment2.docx  # Formatted Step 3 submission template
│
├── data/                                  # Raw and preprocessed dataset files
│   ├── sherlock_raw.txt                   # Project Gutenberg raw text
│   ├── sherlock_clean.txt                 # Normalized text corpus
│   ├── vocab.json                         # Vocabulary word-to-index mapping
│   └── metadata.json                      # Corpus statistics & split details
│
├── src/                                   # Modular source code
│   ├── __init__.py
│   ├── dataset.py                         # Cleaning, tokenization, DataLoader
│   ├── models.py                          # SimpleRNN, BiRNN, LSTM, GRU modules
│   ├── train.py                           # Training loop & metric tracking
│   ├── generate.py                        # Temperature-based autoregressive sampler
│   └── evaluate.py                        # Metrics, distinct n-grams & plotting
│
├── saved_models/                          # Trained model checkpoints (.pt)
│   ├── simple_rnn_best.pt
│   ├── birnn_best.pt
│   ├── lstm_best.pt
│   └── gru_best.pt
│
├── results/                               # Exported evaluation data
│   ├── training_histories.json            # Epoch-by-epoch loss, PPL, acc, time
│   └── generation_results.json            # Temperature-tested generation samples
│
└── plots/                                 # Publication-grade figures
    ├── loss_curves.png
    ├── perplexity_and_accuracy.png
    ├── training_time_and_params.png
    └── diversity_vs_temperature.png
```
