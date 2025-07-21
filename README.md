# EC Enzyme Classifier - Deep Learning for Protein Function Prediction

A PyTorch-based deep learning system for predicting enzyme classification (EC numbers) from protein sequences using ESM-2 protein language model embeddings.

## 🧬 Overview

This project implements a multi-task neural network classifier that predicts the first two levels of EC (Enzyme Commission) numbers from protein amino acid sequences. The model uses state-of-the-art protein embeddings from Meta's ESM-2 (Evolutionary Scale Modeling) transformer model to capture structural and functional information from protein sequences.

### Key Features

- **Protein Language Model Integration**: Uses ESM-2 (8M parameters) to generate contextualized protein embeddings
- **Multi-task Learning**: Simultaneously predicts EC1 and EC2 classification levels
- **Class Imbalance Handling**: Implements weighted BCE loss for imbalanced enzyme classes
- **Comprehensive Evaluation**: Includes F1 scores, ROC curves, confusion matrices, and AUC metrics
- **Long Sequence Support**: Handles sequences up to 1022 amino acids with chunking for longer sequences
- **Production Ready**: Includes inference functions with confidence scoring

## 📊 Architecture

```
Protein Sequence → ESM-2 Embeddings → Dual-Head Classifier → EC1 + EC2 Predictions
     (Raw)           (320-dim)         (Dropout + Linear)      (Multi-class)
```

### Model Components

1. **ESM-2 Encoder**: Converts amino acid sequences to dense vector representations
2. **Dual-Head Classifier**: Two separate linear layers for EC1 and EC2 prediction
3. **Dropout Layer**: Prevents overfitting (p=0.3)
4. **Weighted BCE Loss**: Handles class imbalance with computed class weights

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install matplotlib seaborn
pip install tqdm
pip install fair-esm  # Meta's ESM protein language models
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (A100 recommended for large datasets)
- **Memory**: 16GB+ RAM, 8GB+ VRAM
- **Storage**: 2GB+ for model weights and embeddings

### Data Format

Your training data should be a CSV file with these columns:
- `Sequence`: Amino acid sequences (single letter code)
- `ec_list`: List of EC numbers in string format (e.g., "['1.2.3.4', '2.1.4.5']")

Example:
```csv
Sequence,ec_list
MKLLVLSLCFLATALALAQSACTLQSETH,"['3.2.1.17']"
MYRKLAVISAFLATARAQSACTLQSETHPPL,"['1.1.1.1', '2.3.1.12']"
```

## 🧹 Data Preprocessing

The training pipeline includes essential preprocessing steps to clean and standardize the data:

### Sequence Cleaning
1. **Replace Non-Standard Amino Acids**: Convert "BJOUZ" characters to "X" (unknown amino acid)
   ```python
   # Replace ambiguous amino acids with unknown
   sequence = sequence.replace('B', 'X').replace('J', 'X').replace('O', 'X')
   sequence = sequence.replace('U', 'X').replace('Z', 'X')
   ```

2. **Remove Non-Alphabetic Characters**: Strip numbers, spaces, and special characters
   ```python
   import re
   sequence = re.sub(r'[^A-Za-z]', '', sequence)
   ```

### EC Number Processing
3. **Truncate EC Numbers**: Keep only first 2 digits (EC1.EC2 format)
   ```python
   # From "3.2.1.17" → "3.2"
   ec_truncated = '.'.join(ec_full.split('.')[:2])
   ```

4. **Filter Invalid EC Numbers**: Remove entries containing "-" (incomplete annotations)
   ```python
   # Remove entries like "1.-", "-.2", or "1.2.-"
   valid_ec = [ec for ec in ec_list if '-' not in ec]
   ```

### Deduplication Strategy
5. **Sequence-Based Grouping**: Merge identical sequences with different EC annotations
   ```python
   # Group by sequence, combine EC lists
   # Before: 
   # MKLLVL... → ['1.1.1.1']
   # MKLLVL... → ['2.3.1.12']
   # 
   # After:
   # MKLLVL... → ['1.1.1.1', '2.3.1.12']
   ```

### Complete Preprocessing Example

```python
import pandas as pd
import re
import ast

def preprocess_data(df):
    """Complete preprocessing pipeline"""
    processed_data = []
    
    for _, row in df.iterrows():
        # 1. Clean sequence
        sequence = str(row['Sequence']).upper()
        sequence = sequence.replace('B', 'X').replace('J', 'X').replace('O', 'X')
        sequence = sequence.replace('U', 'X').replace('Z', 'X')
        sequence = re.sub(r'[^A-Za-z]', '', sequence)
        
        # 2. Process EC numbers
        if pd.notnull(row['ec_list']):
            ec_list = ast.literal_eval(row['ec_list'])
            # Truncate to 2 digits and filter out invalid ones
            processed_ec = []
            for ec in ec_list:
                ec_parts = str(ec).split('.')
                if len(ec_parts) >= 2 and '-' not in ec:
                    processed_ec.append(f"{ec_parts[0]}.{ec_parts[1]}")
            
            if processed_ec and sequence:  # Only keep if valid data
                processed_data.append({
                    'Sequence': sequence,
                    'ec_list': processed_ec
                })
    
    # 3. Deduplicate and group by sequence
    sequence_groups = {}
    for item in processed_data:
        seq = item['Sequence']
        if seq in sequence_groups:
            # Merge EC lists and remove duplicates
            combined_ec = list(set(sequence_groups[seq]['ec_list'] + item['ec_list']))
            sequence_groups[seq]['ec_list'] = combined_ec
        else:
            sequence_groups[seq] = item
    
    return pd.DataFrame(list(sequence_groups.values()))

# Apply preprocessing
df_clean = preprocess_data(df_raw)
```

### Preprocessing Impact

- **Sequence Quality**: Ensures only valid amino acid sequences
- **Consistency**: Standardizes ambiguous residue representation
- **EC Standardization**: Focuses on higher-level enzyme classification
- **Data Efficiency**: Eliminates redundant sequences while preserving all annotations
- **Model Performance**: Clean data leads to better training convergence

## 💻 Usage

### Basic Training

```python
# Load and prepare data
df = pd.read_csv("your_training_data.csv")
df['ec_list'] = df['ec_list'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Run the complete training pipeline
python ec_classifier.py
```

### Making Predictions

```python
# Load trained model
checkpoint = torch.load("enhanced_ec2_classifier.pt")
model.load_state_dict(checkpoint['model_state_dict'])
encoders = checkpoint['encoders']

# Predict on new sequence
new_sequence = "MKLLVLSLCFLATALALAQSACTLQSETHPPL..."
predicted_ec, conf1, conf2 = embed_and_predict_full_sequence(
    new_sequence, model, encoders, batch_converter, esm_model, device
)

print(f"Predicted EC: {predicted_ec}")
print(f"Confidence: {(conf1 + conf2) / 2:.4f}")
```

## 📈 Performance Metrics

The model tracks comprehensive performance metrics:

### F1 Score Variants
- **Total F1 (Macro)**: Sum of F1 scores for EC1 and EC2 levels
- **Total F1 (Micro)**: Micro-averaged F1 across both levels
- **Total F1 (Weighted)**: Weighted F1 considering class frequencies

### Additional Metrics
- **ROC AUC**: Area under ROC curve for multi-class classification
- **Confusion Matrices**: Per-class prediction accuracy
- **Classification Reports**: Precision, recall, and F1 per class

### Visualization Outputs

The training process generates several plots:
1. **Training History**: Loss and F1 scores over epochs
2. **ROC Curves**: Multi-class ROC analysis for both EC levels
3. **Confusion Matrices**: Prediction accuracy heatmaps
4. **Performance Comparison**: Side-by-side metric comparisons

## 🔬 Technical Details

### Embedding Generation

The ESM-2 model creates protein embeddings through:
1. **Tokenization**: Converts amino acids to numerical tokens
2. **Transformer Encoding**: 6-layer transformer processes sequence
3. **CLS Token Extraction**: Uses [CLS] token as sequence representation
4. **Dimensionality**: 320-dimensional embeddings per sequence

### Loss Function

Uses Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss):
```python
def bce_digit_loss(logits1, logits2, targets1, targets2):
    # Convert to one-hot encoding
    targets1_oh = F.one_hot(targets1, num_classes=n_ec1).float()
    targets2_oh = F.one_hot(targets2, num_classes=n_ec2).float()
    
    # Apply class weights for imbalance
    bce1 = F.binary_cross_entropy_with_logits(logits1, targets1_oh, reduction='none')
    bce2 = F.binary_cross_entropy_with_logits(logits2, targets2_oh, reduction='none')
    
    # Weighted loss
    weights1 = ec1_weights[targets1].unsqueeze(1)
    weights2 = ec2_weights[targets2].unsqueeze(1)
    
    return (bce1 * weights1).mean() + (bce2 * weights2).mean()
```

### Sequence Length Handling

- **Maximum Length**: 1022 amino acids (ESM-2 limit)
- **Long Sequences**: Chunked with 128 amino acid overlap
- **Embedding Aggregation**: Average pooling across chunks

## 📁 Output Files

After training, the following files are generated:

- `enhanced_ec2_classifier.pt`: Complete model checkpoint including:
  - Model state dictionary
  - Label encoders
  - Training/validation history
  - Final metrics
  - Model configuration

## 🔧 Configuration Options

### Key Hyperparameters

```python
# Model parameters
MAX_LEN = 1022          # Maximum sequence length
BATCH_SIZE = 128        # Training batch size
LEARNING_RATE = 1e-3    # Initial learning rate
DROPOUT = 0.3           # Dropout probability
EPOCHS = 10             # Training epochs

# ESM-2 model selection
esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()  # 8M parameters
# Alternative: esm2_t12_35M_UR50D() for better accuracy (35M parameters)
```

### Memory Optimization

For limited GPU memory:
1. Reduce batch size: `batch_size=64` or `batch_size=32`
2. Use gradient accumulation
3. Enable mixed precision: `torch.cuda.amp.autocast()`
4. Use smaller ESM-2 variant

## 🧪 Results Interpretation

### Understanding EC Numbers

EC (Enzyme Commission) numbers classify enzymes by reaction type:
- **EC1**: Oxidoreductases (oxidation-reduction reactions)
- **EC2**: Transferases (transfer of functional groups)
- **EC3**: Hydrolases (hydrolysis reactions)
- **EC4**: Lyases (addition/removal of groups to form double bonds)
- **EC5**: Isomerases (rearrangement of atoms)
- **EC6**: Ligases (formation of bonds with ATP hydrolysis)

### Confidence Interpretation

- **High Confidence (>0.8)**: Reliable prediction
- **Medium Confidence (0.5-0.8)**: Reasonable prediction, consider manual review
- **Low Confidence (<0.5)**: Uncertain prediction, requires expert validation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional EC number levels (EC3, EC4)
- Ensemble methods for better accuracy
- Active learning for data-efficient training
- Integration with protein structure information

## 📚 References

1. **ESM-2**: Lin, Z. et al. "Evolutionary-scale prediction of atomic level protein structure with a language model." *Science* 379, 1123-1130 (2023).
2. **EC Classification**: Webb, E.C. "Enzyme Nomenclature 1992: Recommendations of the Nomenclature Committee of the International Union of Biochemistry and Molecular Biology." Academic Press (1992).

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient checkpointing
   - Process sequences in smaller chunks

2. **Long Training Times**:
   - Use mixed precision training
   - Consider distributed training for multiple GPUs
   - Pre-compute and cache embeddings

3. **Poor Performance**:
   - Increase training epochs
   - Adjust class weights
   - Use larger ESM-2 model variant
   - Ensure data quality and balance

### Performance Tips

- **Pre-compute Embeddings**: Save embeddings to disk for faster experimentation
- **Batch Processing**: Process multiple sequences simultaneously
- **Model Ensembling**: Combine predictions from multiple models

## 📄 License

This project is released under the MIT License. ESM-2 model weights are subject to Meta's license terms.

---
