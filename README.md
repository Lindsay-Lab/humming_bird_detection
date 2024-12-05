# humming_bird_detection

Classifying bird observations from wildlife cameras.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for humming_bird_detection
│                         and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── humming_bird_detection                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes humming_bird_detection a Python module
    |
    ├── configs        <- Configuration files
    │   └── config_data.yaml
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    |   └── visualize.py
    |
    └── workflow       <- Scripts to organize script executions
        └── Snakefile
    
```

# Initial Outline
- Adapt state-of-the art computer vision model (e.g. YOLOv11) and finetune its classification to detect (humming)-birds by training on open-access bird datasets (e.g. iNaturalist).
- Apply finetuned bird-detection model on photo recordings of bird feeder cameras, reliably selecting images containing birds while handling challenges like different lighting conditions and simultaneous presence of multiple birds by applying appropriate robust model training techniques, data preprocessing, and data augmentation. There is already a considerable set of manually labeled recordings from the cameras that can used as validation set.  
- Handling post-processing to remove potential double-counting and get only discrete observations (the motion detecting cameras always make three photos with 15s delays), and extract date and time for each observation from the image (written in the image's lower corner).
- Communicating with biologists to make the result data available in a usable format.
- Optional extensions: improve the model further to also recognize the type of bird (particularly only humming bird observation), potentially separated further into male and female, as well as types of behavior.


# 2024-12-05 Meeting Take-aways 
- Set up Roboflow to label images with bounding boxes (@Robin)
    - use precise bird type labels when possible
    - what is the limit for the free software tier?
- Lookup statistics of observation datasets: with/without bird, day/night, total (@Valentina)
- Look at similar bird detection project for reference: https://arxiv.org/pdf/2411.15263
- Recruit students for data labeling (@Valentina, @Grace next semester)
- Recruiting master student for model training implementation (@Grace)

