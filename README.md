# project_mlops

Fake news classifier

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── project_mlops  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

PROJECT DESCRIPTION:

The overall goal of this project is to get familiar with and understand not only the way datasets and algorithms or models work but also to get a grasp of how to setup a proper pipeline using the tools provided in the lectures such as Git, cookiecutter, dvc, docker etc. That way we will be able to have a better understanding of the development process as a whole.
 
For this project we have chosen to develop a robust fake news recognition system using machine learning. Our goal is to create an effective model that can distinguish between real and fake news articles and set it up in a way that it will be user-friendly for people that work on it or want to recreate it.
 
Regarding the framework our code will be stored on GitHub and we will use various libraries from PyTorch. Since this is a text classification problem we also aim to use Transformers and text vectorization techniques by taking advantage of what Hugging Face has to offer. For that purpose, we will use hydra to easily train our models with different parameters such as tokenizers, model architectures, pretrained parameters or hyperparameter tuning settings.  
 
 
Our dataset for this project is the "Fake News Prediction Dataset" that we have found on kaggle. This dataset has a record of fake and real news and it will be used for binary classification "fake" or "real" news.
 
As far as models are concerned we aim to use models that are relevant with Natural Language Processing such as RoBERTa that is a pretrained NLP model. However it is more likely that we will start with something simpler like a Convolutional Neural Network aimed for text classification or some standard algorithms such as Support Vector Machines. Furthermore we aim to compare all to see the differences in overall performance such as time, usage and accuracy.