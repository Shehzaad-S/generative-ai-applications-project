### Generatice AI Applications Project

### Project Description

This project implements a GPT-style decoder-only Transformer from first principles and applies it to conversational language modeling using the Cornell Movie-Dialogs Corpus. Using Python, PyTorch, NumPy, Matplotlib, Hugging Face Tokenizers, and modern Transformer training techniques, the project demonstrates an end-to-end deep learning workflow including corpus reconstruction, exploratory analysis, custom tokenizer development, Transformer implementation, model training, controlled experimentation, and evaluation.

The primary objective of the project is to investigate whether a custom-tokenized decoder-only Transformer can learn meaningful conversational patterns from human-authored movie dialogue while remaining computationally efficient on modest GPU hardware. Unlike a standard GPT implementation, this project places significant emphasis on data-centric design decisions, vocabulary optimization, conversational boundary modeling, and resource-aware experimentation.

---

#### What I Built

- Reconstructed complete conversations from the Cornell Movie-Dialogs Corpus.
- Performed exploratory analysis of conversation lengths, vocabulary utilization, and dialogue structure.
- Developed and evaluated a custom Byte Pair Encoding (BPE) tokenizer.
- Introduced explicit conversational boundary markers (`<|eou|>` and `<|endofdialog|>`).
- Implemented token embeddings and positional embeddings.
- Implemented causal multi-head self-attention using PyTorch's modern attention primitives.
- Developed Transformer blocks using layer normalization, residual connections, GELU activations, feed-forward networks, and dropout.
- Built a GPT-style decoder-only Transformer architecture for autoregressive language generation.
- Developed training, validation, and testing pipelines using PyTorch.
- Implemented mixed precision training (AMP), gradient clipping, checkpointing, and loss monitoring.
- Conducted controlled depth and regularization experiments.
- Evaluated model performance using validation loss, test loss, perplexity, and qualitative generation examples.
- Analyzed trade-offs between tokenizer efficiency, model depth, regularization, and conversational quality.
- Summarized findings, limitations, ethical considerations, and future improvement opportunities.

---

### Dataset

#### Cornell Movie-Dialogs Corpus

Source: Cornell University

Paper: _Chameleons in Imagined Conversations: A New Approach to Understanding Coordination of Linguistic Style in Dialogs_ (Danescu-Niculescu-Mizil & Lee, 2011)

Dataset Link:

https://www.cs.cornell.edu/~cristian/research#paper-chameleons-in-imagined

#### Dataset Characteristics

After reconstruction and preprocessing, the corpus contained:

- 83,097 conversations
- Approximately 3.5 million words
- Diverse movie-style dialogue including:
  - Emotional conversations
  - Interpersonal relationships
  - Dramatic exchanges
  - Narrative interactions
  - Question-answer dialogue

Custom conversation boundary markers were preserved throughout preprocessing to help the model learn conversational turn-taking and dialogue termination behavior.

---

### Notebook Execution Notes

The project was developed locally and trained using Google Colab GPU resources.

A small number of Colab-specific helper commands remain in the notebook as comments for reproducibility, such as:

```python
# %cd /content/deep-learning-systems-project
# !ls -la
```

These commands are not required when running the notebook locally and may be safely ignored.

All analysis, experimental results, generated examples, and discussion are included within the notebook and accompanying report. Reviewers are not required to retrain models to evaluate the project.

---

#### Bias Awareness and Limitations

As with all generative language models, results should be interpreted with appropriate caution.

Potential limitations include:

- The corpus is limited to movie dialogue and does not represent the full diversity of real-world language use.
- Generated responses may contain factual inaccuracies despite appearing coherent.
- Longer conversations occasionally lose coherence or become repetitive.
- The model exhibits limited factual reasoning outside the domain of movie dialogue.
- Training and experimentation were constrained by available GPU resources.
- The model contains no content moderation or safety guardrails.

---

#### Future Extensions

Several realistic avenues for future improvement were identified:

#### Data Improvements

- Incorporate additional conversational datasets.
- Investigate larger and more diverse dialogue corpora.
- Explore multilingual conversational datasets.

#### Model Improvements

- Larger context lengths.
- Additional Transformer depth experiments.
- Alternative attention mechanisms.
- Systematic hyperparameter optimization.
- More advanced regularization techniques.

#### Deployment Improvements

- Safety guardrails and content filtering.
- Interactive chatbot interface for human evaluation.
- Real-time conversational application deployment.

---

### How To Run The Project

#### Model Checkpoints

Model checkpoints are intentionally excluded from GitHub and Git because of storage limitations. All training code, experiments, metrics, and evaluation results remain fully reproducible.

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd generative-ai-applications-project
```

#### 2. Create and Activate an Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate statistical_analysis
```

Using venv:

```bash
python -m venv statistical_analysis
```

Windows:

```bash
statistical_analysis\Scripts\activate.bat
```

Linux/macOS:

```bash
source statistical_analysis/bin/activate
```

#### 3. Install Dependencies

Python Version:

```text
Python 3.11+
```

Using pip:

```bash
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
```

#### 4. Open the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Open:

```text
deep_learning_systems_project.ipynb
```

Run all cells sequentially from top to bottom.

---

### Running With Google Colab

The notebook supports both local execution and GPU-enabled execution within Google Colab.

Typical workflow:

1. Clone the repository within Colab.
2. Verify project paths.
3. Enable a GPU runtime.
4. Install project dependencies.
5. Run notebook cells sequentially.
6. Execute training experiments.

Reviewers may execute the notebook locally without requiring Colab-specific configuration.

---

### Reproducibility

Generate a requirements file:

```bash
pip freeze > requirements.txt
```

Generate a Conda environment specification:

```bash
conda env export > environment.yml
```

The repository includes both `requirements.txt` and `environment.yml` to support environment recreation and reproducible experimentation.
