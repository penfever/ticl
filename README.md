# ticl - Tabular In-Context Learning

This repository contains code for training and prediction of several models for tabular in-context learning, including **MotherNet**, **GAMformer** and **TabFlex**.
**MotherNet** is a hypernetwork foundational model (or conditional neural process) for tabular data classification that creates a small neural network.
**GAMformer** is a model trained to output an interpretable, additive model using in-context learning.
**TabFlex** is a extension of ``TabPFN``  using linear attention that overcomes the scaling limitations of ``TabPFN`` in terms of features, models and number of classes.

- [MotherNet](#MotherNet)
- [GAMformer](#GAMformer)
- [TabFlex](#TabFlex)

Both the architecture and the code in this repository is based on the [TabPFN](https://github.com/automl/TabPFN) by the [Freiburg AutoML group](https://www.automl.org/).

The repository includes code for training and prediction with these models, as well as links to checkpoints for the models used in our publications.

All models provided are research prototypes, shared for research use, and not meant for real-world applications. Responsibility for using the models contained in this repository, as well monitoring and assessing potential impact of the models lies with the user of the code.

# MotherNet

## Installation

It's recommended to use conda to create an environment using the provided environment file:

```
conda create -f environment.yml
```
Then install the package:
```
conda activate ticl
pip install -e .
```

## Getting started

A simple usage of the MotherNet sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ticl.prediction import MotherNetClassifier, EnsembleMeta

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# MotherNetClassifier encapsulates a single instantiation of the model.
# This will automatically download a model from blob storage

classifier = MotherNetClassifier(device='cpu')

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))

# Ensembling as described in the TabPFN paper an be performed using the EnsembleMeta wrapper
ensemble_classifier = EnsembleMeta(classifier)
# ...
```

### MotherNet Usage

MotherNet uses the same preprocessing as the TabPFN work it builds upon, but we found that using one-hot-encoding during inference improves accuracy.
Scaling of features is handled internally.

## Model Training

This repository provides extensive functionality for training various tabular in-context learning models. The training process is managed through the `fit_model.py` script, which offers a wide range of configuration options.

### Training Framework

Training models in this repository follows a consistent pattern:

```bash
python -m ticl.fit_model <model_type> [options]
```

Where `<model_type>` is the model architecture you want to train. The framework supports multiple model types:

- `mothernet` - MotherNet hypernetwork foundational model
- `tabflex` - TabFlex model with linear attention mechanism
- `additive` - Additive MotherNet model
- `baam` - Bi-attention additive MotherNet model
- `perceiver` - Perceiver variant of MotherNet
- `tabpfn` - Original TabPFN model
- `batabpfn` - Bi-attention TabPFN model
- `la_mothernet` - Linear attention MotherNet variant

### Common Training Options

All models share common configuration options:

#### General Options
- `-g, --gpu-id` - Specify GPU ID to use for training
- `-C, --use-cpu` - Use CPU instead of GPU for training

#### Optimizer Options
- `-E, --epochs` - Number of training epochs
- `-l, --learning-rate` - Maximum learning rate
- `-k, --aggregate_k_gradients` - Number of steps to aggregate gradient over
- `-A, --adaptive-batch-size` - Whether to progressively increase effective batch size
- `-w, --weight-decay` - Weight decay for AdamW optimizer
- `-Q, --learning-rate-schedule` - Learning rate schedule (cosine, constant, exponential)
- `-U, --warmup-epochs` - Number of epochs to warm up learning rate
- `-t, --train-mixed-precision` - Whether to train with mixed precision

#### Dataloader Options
- `-b, --batch-size` - Physical batch size
- `-n, --num-steps` - Number of steps per epoch
- `--min-eval-pos` - Minimum evaluation position
- `--random-n-samples` - Whether to sample n_samples randomly
- `--n-test-samples` - Number of test samples

#### Transformer/Attention Options
- `-e, --emsize` - Embedding size
- `-N, --nlayers` - Number of transformer/attention layers
- `--init-method` - Weight initialization method
- `--y-encoder` - Encoder for labels (linear, onehot, or None)
- `--classification-task` - Whether to use classification or regression

#### Prior and Data Generation
- `--num-features` - Maximum number of features in prior
- `--n-samples` - Maximum number of samples in prior
- `--prior-type` - Prior type to use (prior_bag, boolean_only, bag_boolean, step_function)

#### Orchestration and Logging
- `--use-mlflow` - Enable MLFlow tracking (requires MLFLOW_HOSTNAME environment variable)
- `--use-wandb` - Enable Weights & Biases tracking
- `--save-every` - Save model every N epochs
- `-f, --warm-start-from` - Warm start from a checkpoint file
- `-c, --continue-run` - Continue from a previous run

### Model-Specific Options

Different model types have additional specific configuration options:

#### MotherNet Options (`mothernet`, `la_mothernet`)
- `-d, --decoder-embed-dim` - Decoder embedding size
- `-H, --decoder-hidden-size` - Decoder hidden size
- `--decoder-activation` - Decoder activation function
- `-D, --decoder-type` - Decoder type (output_attention, special_token, class_average, average)
- `-T, --decoder-hidden-layers` - Number of hidden layers in decoder MLP
- `-P, --predicted-hidden-layer-size` - Size of hidden layers in predicted network
- `-L, --predicted-hidden-layers` - Number of predicted hidden layers
- `--predicted-activation` - Activation in predicted network
- `-r, --low-rank-weights` - Whether to use low-rank weights in mothernet
- `-W, --weight-embedding-rank` - Rank of weights in predicted network

#### Additive Model Options (`additive`, `baam`)
- `--input-bin-embedding` - Bin embedding type (linear, non-linear, none)
- `--bin-embedding-rank` - Rank of bin embedding
- `--fourier-features` - Number of Fourier features to add per feature
- `--n-bins` - Number of bins
- `--nan-bin` - Whether to use the last bin to denote a nan value
- `--categorical-embedding` - Whether to embed categorical features using a separate embedding
- `--marginal-residual` - Whether to learn the residual of the marginals

#### TabFlex Options
- `--model` - Which linear attention model to use (linear_attention, fla)
- `--feature-map` - Feature map to use with FLA (identity_for_real, elu, hedgehog, hedgehog_shared)
- `--norm-output` - Whether to normalize the output of the model
- `--causal-mask` - Whether to use causal attention

### Training Examples

Here are examples for training different model types:

#### Training TabPFN

The default hyperparameters are found in ticl/model_configs.py in the
   get_tabpfn_default_config() function, which inherits from
  get_shared_defaults(). Key parameters include:

  - Transformer: emsize=512, nlayers=12, nhead=4, y_encoder="one_hot",
  classification_task=True
  - Optimizer: learning_rate=0.00003, epochs=4000, warmup_epochs=20
  - Prior/Data: num_features=100, n_samples=1152, batch_size=8,
  num_steps=8192, prior-type=prior_bag

```bash
python -m ticl.fit_model tabpfn -h
```

#### Training MotherNet

Basic training with default parameters:
```bash
python -m ticl.fit_model mothernet
```

Training with custom parameters:
```bash
python -m ticl.fit_model mothernet -g 0 -E 2000 -l 0.0001 -b 16 -L 2 -P 256
```

The settings used in the MotherNet paper:
```bash
python -m ticl.fit_model mothernet -L 2
```

#### Training TabFlex

Basic training:
```bash
python -m ticl.fit_model tabflex
```

With custom parameters:
```bash
python -m ticl.fit_model tabflex -g 0 -E 3000 -l 0.00005 -b 16 --model linear_attention
```

#### Training Additive Model

```bash
python -m ticl.fit_model additive -g 0 -E 2000 -n-bins 64 --categorical-embedding True
```

### Multi-GPU Training

Data-parallel multi-GPU training is supported using `torchrun`:

```bash
torchrun --nproc_per_node=<num_gpus> -m ticl.fit_model <model_type> [options]
```

### Experiment Tracking

The framework supports two experiment tracking platforms: MLFlow and Weights & Biases (wandb).

#### MLFlow Integration

MLFlow tracking is enabled with the `--use-mlflow` flag. You need to set the `MLFLOW_HOSTNAME` environment variable to specify the MLFlow server:

```bash
export MLFLOW_HOSTNAME=localhost  # or your MLFlow server address
python -m ticl.fit_model mothernet --use-mlflow
```

MLFlow configuration options:
- `--experiment` - Name of the MLFlow experiment (default: 'Default')
- `-R, --create-new-run` - Create a new MLFlow run even when continuing training
- `--save-every` - Save checkpoints every N epochs (default: 10)

Example with custom experiment name:
```bash
python -m ticl.fit_model mothernet --use-mlflow --experiment "MotherNet-Experiments"
```

#### Weights & Biases (wandb) Integration

Weights & Biases tracking is enabled with the `--use-wandb` flag:

```bash
python -m ticl.fit_model mothernet --use-wandb
```

wandb configuration options:
- `--wandb-overwrite` - Whether to overwrite existing wandb runs (default: False)

The wandb project, entity, and directory are configured in the `environment.py` file. To customize these settings, you should edit this file:

```python
# ticl/environment.py
WANDB_INFO = {
    "project": "your-project-name",  # Change to your project name
    "entity": "your-username",       # Change to your wandb username or team
    "dir": './wandb',                # Directory for wandb files
}
```

Alternatively, you can set wandb parameters using environment variables:
```bash
export WANDB_PROJECT="your-project-name"
export WANDB_ENTITY="your-username"
export WANDB_DIR="./custom-wandb-dir"
python -m ticl.fit_model mothernet --use-wandb
```

#### Using Both Tracking Systems

You can use both MLFlow and wandb simultaneously:

```bash
export MLFLOW_HOSTNAME=localhost
python -m ticl.fit_model mothernet --use-mlflow --use-wandb --experiment "Dual-Tracking-Experiment"
```

#### Continuing Experiments

To continue training from a checkpoint with experiment tracking:

```bash
python -m ticl.fit_model mothernet -f /path/to/checkpoint.pt -c --use-mlflow
```

If you want to create a new run in MLFlow even when continuing:
```bash
python -m ticl.fit_model mothernet -f /path/to/checkpoint.pt -c -R --use-mlflow
```

## Understanding Priors

Training models in this repository relies on synthetic data generation using various "priors" - different ways of generating synthetic tabular datasets with known properties. These priors help the model learn generalizable patterns for in-context learning. The repository includes several prior types with different characteristics:

### Prior Types

#### StepFunctionPrior
A simple prior that generates data using randomly placed step functions. Features are randomly generated, and a small number of features are selected to create decision boundaries. The target variable is determined by whether input values are greater than or less than the corresponding step boundary.

**Parameters:**
- `max_steps`: Maximum number of steps in the step function (default: 1)
- `sampling`: Data sampling strategy - 'uniform' or 'normal'

**Example Usage:**
```python
from ticl.priors import StepFunctionPrior
config = {'max_steps': 2, 'sampling': 'uniform'}
prior = StepFunctionPrior(config)
```

#### MLPPrior
Generates synthetic data using a randomly initialized multi-layer perceptron (MLP). This prior creates complex nonlinear relationships between features and targets, simulating real-world tabular data with intricate patterns.

**Parameters:**
- `num_layers`: Number of hidden layers in the MLP
- `prior_mlp_hidden_dim`: Hidden dimension size in the MLP
- `prior_mlp_activations`: Activation functions to use (e.g., ReLU, Tanh)
- `noise_std`: Standard deviation of noise added to outputs
- `init_std`: Standard deviation for weight initialization
- `prior_mlp_dropout_prob`: Dropout probability for MLP weights
- `is_causal`: Whether to use a causal structure
- `num_causes`: Number of causal variables if using causal structure
- `add_uninformative_features`: Whether to add uninformative features
- `sampling`: Data sampling strategy - 'normal', 'mixed', or 'uniform'

**Example Usage:**
```python
from ticl.priors import MLPPrior
config = {
    'num_layers': 3, 
    'prior_mlp_hidden_dim': 128,
    'noise_std': 0.1,
    'sampling': 'normal'
}
prior = MLPPrior(config)
```

#### BooleanConjunctionPrior
Generates data based on boolean conjunctions (AND operations) with random feature selection. This prior creates logical rule-based datasets where the target depends on logical combinations of binary features.

**Parameters:**
- `max_rank`: Maximum number of features to include in each conjunction
- `max_fraction_uninformative`: Maximum fraction of uninformative features
- `p_uninformative`: Probability of adding uninformative features

**Example Usage:**
```python
from ticl.priors import BooleanConjunctionPrior
config = {
    'max_rank': 5,
    'max_fraction_uninformative': 0.3,
    'p_uninformative': 0.2
}
prior = BooleanConjunctionPrior(config)
```

#### GPPrior
Generates synthetic data using Gaussian Process regression with configurable kernels. This prior creates smooth, continuous functional relationships between inputs and targets.

**Parameters:**
- `outputscale`: Output scale parameter for the RBF kernel
- `lengthscale`: Length scale parameter for the RBF kernel
- `noise`: Noise level in the Gaussian likelihood
- `sampling`: Data sampling strategy - 'uniform' or 'normal'

**Example Usage:**
```python
from ticl.priors import GPPrior
config = {
    'outputscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
    'lengthscale': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 8},
    'noise': 0.01,
    'sampling': 'normal'
}
prior = GPPrior(config)
```

#### BagPrior
Combines multiple prior types into a single prior by sampling from them according to specified weights. This allows training on a mixture of different data-generating processes.

**Parameters:**
- `base_priors`: Dictionary of prior objects to sample from
- `prior_weights`: Dictionary of weights for each prior

**Example Usage:**
```python
from ticl.priors import BagPrior, MLPPrior, StepFunctionPrior
mlp_prior = MLPPrior({'num_layers': 3, 'sampling': 'normal'})
step_prior = StepFunctionPrior({'max_steps': 2})
bag_prior = BagPrior(
    base_priors={'mlp': mlp_prior, 'step': step_prior},
    prior_weights={'mlp': 0.7, 'step': 0.3}
)
```

NOTE: This is currently hard-coded to MLP, GP, heavily favoring MLP

```python
prior = BagPrior(base_priors={'gp': gp_flexible, 'mlp': mlp_flexible},
                         prior_weights={'mlp': 0.961, 'gp': 0.039})
```

#### ClassificationAdapterPrior
Wraps other priors to adapt them for classification tasks. It handles creating class boundaries, balancing classes, adding missing values (NaNs), and categorical features.

**Parameters:**
- `base_prior`: The prior to wrap
- `max_num_classes`: Maximum number of classes (0 means regression)
- `num_classes`: Number of classes to use for a specific batch
- `multiclass_type`: Type of multiclass boundary ('steps' or 'rank')
- `nan_prob_no_reason`: Probability of inserting NaNs randomly
- `nan_prob_a_reason`: Probability of inserting NaNs with correlation to features
- `categorical_feature_p`: Probability of converting features to categorical
- `pad_zeros`: Whether to pad with zeros for consistent feature count
- `feature_curriculum`: Whether to use curriculum learning for features

**Example Usage:**
```python
from ticl.priors import ClassificationAdapterPrior, MLPPrior
base_prior = MLPPrior({'num_layers': 3})
config = {
    'max_num_classes': 10,
    'multiclass_type': 'rank',
    'categorical_feature_p': 0.2,
    'nan_prob_no_reason': 0.05,
    'feature_curriculum': False
}
prior = ClassificationAdapterPrior(base_prior, **config)
```

### Using Priors in Training

When training models, the prior type is specified using the `--prior-type` command line argument. The most common option is `prior_bag`, which combines multiple priors:

```bash
python -m ticl.fit_model mothernet --prior-type prior_bag
```

For single prior types:

```bash
python -m ticl.fit_model mothernet --prior-type step_function
python -m ticl.fit_model mothernet --prior-type boolean_only
```

Prior hyperparameters can be configured through various command line arguments under the `prior` group. For example:

```bash
python -m ticl.fit_model mothernet --prior-type prior_bag --num-features 100 --n-samples 1024
```

The distribution of priors in the bag is configured in the model configuration file, with weights specified for each prior type.

## Papers
This work is described in [MotherNet: A Foundational Hypernetwork for Tabular Classification](https://arxiv.org/pdf/2312.08598).
Please cite that work when using this code. As this work rests on the TabPFN work, I would suggest you also cite their [paper](https://arxiv.org/abs/2207.01848),
which also provides more background on the methodology.

# GAMformer

WIP

# TabFlex

Recent advances in the field of in-context learning (ICL) have demonstrated impressive performance for tabular classification, exemplified by TabPFN's success on small datasets. However, the quadratic complexity of the attention mechanism limits its applicability to larger datasets. To address this issue, we conduct a comprehensive comparison of popular scalable attention alternatives, including state-space models (SSMs) and linear attention mechanisms, revealing that the inherent causality of SSMs hinders ICL performance for large datasets, while linear attention preserves effectiveness. Leveraging these insights, we introduce TabFlex, a model based on linear attention that supports thousands of features and hundreds of classes, capable of handling datasets with millions of samples. Extensive experiments demonstrate that TabFlex is significantly faster than most existing methods while achieving top-two performance on small datasets among 25 baselines, with a 2xspeedup over TabPFN and a 1.5xspeedup over XGBoost. On large datasets, TabFlex remains efficient (e.g., approximately 5 seconds on the poker-hand dataset, which consists of millions of samples), while achieving relatively solid performance.

---

## **Step 1: Install Environment for TabFlex**

Create the Conda environment using the provided file:

   ```bash
   git clone https://github.com/microsoft/ticl
   conda env create -f ticl/tabflex_conda.yaml
   cd ../ticl
   pip install -e .
   ```

---

## **Step 2: Model Inference**

Below is an example of using TabFlex for logistic classification.

```python
from ticl.prediction.tabflex import TabFlex
import torch

# Generate synthetic dataset
X_train = torch.randn(300, 20)
coef = torch.randn(20) 
y_train = (X_train @ coef > 0).int()

X_test = torch.randn(50, 20)
y_test = (X_test @ coef > 0).int()

# Initialize and train TabFlex model
tabflex = TabFlex()
tabflex.fit(X_train, y_train)

# Make predictions
y_pred = tabflex.predict(X_test)

# Evaluate performance
acc = (torch.tensor(y_pred) == y_test).float().mean().item()
print(f"Accuracy: {acc:.4f}")
```

---

## **Step 3: Test TabFlex on Different Datasets**

To evaluate TabFlex on various datasets, use [TabZilla](https://github.com/yzeng58/tabzilla). Follow these steps:

1. Clone the TabZilla repository:
   ```bash
   git clone https://github.com/yzeng58/tabzilla
   ```
2. Follow the instructions in the TabZilla README.  
   - When specifying the `--model_name` parameter, set it to `tabflex`:
     ```bash
     --model_name tabflex
     ```

---



## License
Copyright 2022 Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter

Additions by Andreas Mueller, 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
