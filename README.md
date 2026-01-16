<div align="center">

# A Meta-Predictor Framework for Proactive Defense Against Transferable Jailbreak Attacks in Large Language Models

This framework is used to quickly predict similarity metrics between different large models to defend against transferable jailbreak attacks in large language models. It supports calculating 15 sets of metrics for combinations of 3 source models and 5 target models respectively, and uniformly records the metric results of each category together.

## Framework Overview

It contains two main Python scripts:

1. **model_similarity_analysis.py** - The full version, which implements the calculation of all three types of similarity metrics and provides detailed visualization and report generation functions.
2. **test_model_similarity.py** - A test script used to verify whether the main functions work properly.

## Categories of Similarity Metrics

According to the documentation, the similarity metrics are divided into the following three categories:

### 1. Output Distribution Similarity Metrics
- KL Divergence: Measures the degree of difference between the output distribution of one model and that of another model.
- Jensen-Shannon Divergence (JSD): A symmetric and smoothed version of KL, with a value range of [0,1].
- Earth Mover's Distance (EMD): The minimum "transportation cost" to transform one distribution into another.
- Logits Cosine Similarity: Measures the angle between output vectors, which is simple and scale-independent.
- RBO (Rank-Based Overlap): Measures the degree of overlap between the top k tokens.

### 2. Representation Space Similarity Metrics
- Centered Kernel Alignment (CKA): Compares the similarity between two sets of embedding representations.
- SVCCA (Singular Value Canonical Correlation Analysis): Finds highly correlated subspaces in two representation spaces.
- PWCCA (Weighted SVCCA): weights SVCCA to emphasize important directions.
- RSA (Representational Similarity Analysis): Compares the distance patterns between pairs of samples in the embedding space.

### 3. Behavioral/Functional Similarity Metrics
- Task Consistency Rate: The proportion of cases where the two models have the same top-1 output for the same input.
- Pass@k Consistency Rate: The probability that both models include the correct/target answer in their top-k outputs.
- Adversarial Transfer Rate: The proportion of cases where a jailbreak prompt targeting model A also succeeds on model B.
- Semantic Similarity: Calculates the semantic cosine similarity of the outputs of two models using sentence vectors.

## Supported Model Combinations

The tool supports calculating 15 combinations of the following 3 source models and 5 target models:

**Source Models:**
- llama2-7b
- bert-large
- roberta-large

**Target Models:**
- mistral-7b
- vicuna-7b
- guanaco-7b
- starling-7b
- chatgpt-3.5

## Usage Methods

### Environment Requirements

- Python 3.6+ (3.8+ recommended)
- Required Python libraries: numpy, matplotlib

You can install the necessary libraries using the following command:

```bash
pip install numpy matplotlib
```

### Running the Scripts

#### Running the Complete Calculation for 15 Model Pairs

```bash
python model_similarity_analysis.py
```

#### Running the Test Script

If you want to test the functionality first without running the complete calculation:

```bash
python test_model_similarity.py
```

## Output Results

### Complete Calculation Output

After running the full version, a folder named `similarity_analysis_results` will be created in the current directory, containing the following contents:

1. **JSON format result files**: Containing detailed calculation results of all metrics.
2. **Classification result files**: Results are saved separately by metric category (output distribution, representation space, behavioral/functional), and the metric results of each category are uniformly recorded together.
3. **Visualization charts**: Heatmaps of various metrics to intuitively show the similarity between different model pairs.
4. **Summary report**: Including calculation time, model information, statistical summaries of various metrics, etc.

### Test Script Output

After running the test script, a folder named `test_similarity_results` will be created in the current directory, containing the calculation results of a single pair of models, which is used to verify whether the functionality is normal.

## Detailed Explanation of Script Functions

### Main Functions of model_similarity_analysis.py

- **Data Generation**: Generates simulated data such as model output probability distributions, logits, and hidden layer representations.
- **Metric Calculation**: Implements the calculation of 12 similarity metrics in all three categories.
- **Result Saving**: Saves results in JSON format, recorded separately by metric category.
- **Visualization**: Creates heatmaps to intuitively display calculation results.
- **Report Generation**: Generates a detailed summary report containing statistical information of various metrics.

### Main Functions of test_model_similarity.py

- **Instance Verification**: Verifies whether the ModelSimilarityAnalysis class can be initialized correctly.
- **Model List Verification**: Checks whether the configuration of source models and target models is correct.
- **Single Pair Calculation Test**: Tests whether the function of calculating metrics for a pair of models works properly.
- **Result Structure Check**: Verifies whether the data structure of the calculation results meets expectations.
- **Result Saving Test**: Tests whether the result saving function works properly.

## Custom Configuration

### Adjusting the Model List

If you need to adjust the list of source models or target models, you can modify the following parts in the script:

```python
# Source model list
source_models = ['llama2-7b', 'bert-large', 'roberta-large']
# Target model list
target_models = ['mistral-7b', 'vicuna-7b', 'guanaco-7b', 'starling-7b', 'chatgpt-3.5']
```

### Adjusting the Output Directory

You can customize the output directory by modifying the initialization parameters:

```python
# Modify the output directory
metrics_calculator = ModelSimilarityAnalysis(output_dir="./custom_results")
```
