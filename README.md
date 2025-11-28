# Generated Sentence Templates using GANs and spaCy

This project generates sentence templates using Generative Adversarial Networks (GANs) trained on linguistic features extracted from "The Wizard of Oz" novel using spaCy. Instead of generating actual words, the model learns to generate grammatical structures in the form of Part-of-Speech (POS) tags and dependency relations.

## Project Overview

```mermaid
flowchart TD
    A[The Wizard of Oz Text] --> B[Text Preprocessing]
    B --> C[Sentence Segmentation]
    C --> D[spaCy NLP Processing]
    D --> E[Extract POS Tags]
    D --> F[Extract Dependencies]
    E --> G[Create Lookup Dictionaries]
    F --> G
    G --> H[Vectorization]
    H --> I[Fixed-Length Vectors<br/>128 dimensions]
    I --> J[GAN Training]
    J --> K[Generator Model]
    J --> L[Discriminator Model]
    K --> M[Generate New Templates]
    L --> M
    M --> N[Convert to POS/Dependency Sequences]
```

## Detailed Workflow

### 1. Data Preprocessing Pipeline

```mermaid
flowchart LR
    A[Raw Text File] --> B[Read & Clean Text]
    B --> C[Remove Special Characters]
    C --> D[Split into Sentences]
    D --> E[For Each Sentence]
    E --> F[Preprocess Text]
    F --> G[spaCy Processing]
    G --> H[Extract POS Tags]
    G --> I[Extract Dependencies]
    H --> J[POS Tag List]
    I --> K[Dependency List]
    J --> L[Combine Results]
    K --> L
```

### 2. Vectorization Process

```mermaid
flowchart TD
    A[POS Tags List] --> B[Create POS Lookup Dictionary]
    C[Dependencies List] --> D[Create Dependency Lookup Dictionary]
    B --> E[For Each Sentence]
    D --> E
    E --> F[Convert to Indices]
    F --> G[Combine: 64 deps + 64 POS]
    G --> H[Fixed-Length Vector<br/>128 dimensions]
    H --> I[Zero Padding if needed]
    I --> J[Training Dataset]
```

### 3. GAN Architecture

```mermaid
graph TB
    subgraph Generator
        Z[Random Noise<br/>60 dims] --> G1[Linear Layer<br/>60→16]
        G1 --> G2[LeakyReLU + Dropout]
        G2 --> G3[Linear Layer<br/>16→16]
        G3 --> G4[LeakyReLU + Dropout]
        G4 --> G5[Linear Layer<br/>16→128]
        G5 --> G6[Tanh Activation]
        G6 --> GOUT[Generated Vector<br/>128 dims]
    end
    
    subgraph Discriminator
        INPUT[Input Vector<br/>128 dims] --> D1[Linear Layer<br/>128→16]
        D1 --> D2[LeakyReLU + Dropout]
        D2 --> D3[Linear Layer<br/>16→16]
        D3 --> D4[LeakyReLU + Dropout]
        D4 --> D5[Linear Layer<br/>16→1]
        D5 --> DOUT[Real/Fake Score]
    end
    
    GOUT --> INPUT
```

### 4. Training Process

```mermaid
flowchart TD
    A[Start Training] --> B[For Each Epoch]
    B --> C[For Each Batch]
    C --> D[Train Discriminator]
    D --> D1[Real Data → D_real_loss]
    D --> D2[Fake Data from Generator → D_fake_loss]
    D1 --> D3[Backpropagate D_loss]
    D2 --> D3
    D3 --> E[Train Generator]
    E --> E1[Generate Fake Data]
    E1 --> E2[Discriminator Evaluation]
    E2 --> E3[Use Real Loss<br/>Flip Labels]
    E3 --> E4[Backpropagate G_loss]
    E4 --> F{More Batches?}
    F -->|Yes| C
    F -->|No| G{More Epochs?}
    G -->|Yes| B
    G -->|No| H[Training Complete]
```

### 5. Generation Process

```mermaid
flowchart TD
    A[Trained Generator] --> B[Sample Random Noise<br/>60 dimensions]
    B --> C[Generate Vector<br/>128 dimensions]
    C --> D[Unnormalize Values]
    D --> E[Split Vector<br/>64 deps + 64 POS]
    E --> F[Lookup Dependencies]
    E --> G[Lookup POS Tags]
    F --> H[Generated Dependency Sequence]
    G --> I[Generated POS Sequence]
    H --> J[Output Template]
    I --> J
```

## Architecture Details

### Model Specifications

- **Input Vector Size**: 128 dimensions (64 dependencies + 64 POS tags)
- **Sentence Length**: Fixed at 64 tokens
- **Generator**:
  - Input: 60-dimensional noise vector
  - Hidden Layers: 2 fully connected layers (16 neurons each)
  - Output: 128-dimensional vector
  - Activation: Tanh
  
- **Discriminator**:
  - Input: 128-dimensional vector
  - Hidden Layers: 2 fully connected layers (16 neurons each)
  - Output: 1-dimensional score (real/fake)
  - Activation: LeakyReLU (0.2)

### Training Parameters

- **Epochs**: 20
- **Batch Size**: 32
- **Learning Rate**: 0.00002
- **Optimizer**: Adam
- **Dropout**: 0.3

## Key Features

1. **Linguistic Feature Extraction**: Uses spaCy to extract grammatical structures rather than raw text
2. **Template Generation**: Generates sentence templates (POS + dependencies) instead of actual words
3. **Fixed-Length Encoding**: All sentences converted to uniform 128-dimensional vectors
4. **GAN-based Learning**: Adversarial training to learn realistic sentence structures

## Dependencies

- spaCy (`en_core_web_sm` model)
- PyTorch
- NumPy
- scikit-learn
- Google Colab (for file mounting)

## Usage

The project is implemented as a Jupyter notebook (`576_Spacy_Vector_GANsipynb.ipynb`). Run the cells sequentially to:

1. Mount Google Drive and load the text file
2. Process text with spaCy
3. Create vectorized training data
4. Train the GAN
5. Generate new sentence templates

## Output

The model generates sequences of:
- **Dependency Relations**: e.g., `['ROOT', 'nsubj', 'dobj', 'prep', ...]`
- **POS Tags**: e.g., `['NOUN', 'VERB', 'DET', 'ADJ', ...]`

These can be used as templates for generating grammatically structured sentences.
