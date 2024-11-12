# <b>which bin?: Intelligent garbage sorting assistance</b>

<b>which bin?</b> is a web application designed to assist users in determining the correct disposal bin for their garbage based on local sorting guidelines. The app leverages machine learning to classify garbage images into one of several categories (plastic, glass, metal, clothes etc.) and provides tailored disposal instructions according to the user's selected city. This project started as the final project for the ML for business 2 course given by Mathieu Soul at Albert School Paris.

## <b>Table of contents</b>

- [which bin?: Intelligent garbage sorting assistance](#which-bin-intelligent-garbage-sorting-assistance)
  - [Table of contents](#table-of-contents)
  - [Project overview](#project-overview)
  - [Features](#features)
  - [Project structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
  - [Model training and evaluation](#model-training-and-evaluation)
    - [Dataset Description](#dataset-description)
    - [Baseline Model](#baseline-model)
    - [First iteration](#first-iteration)
    - [Second iteration](#second-iteration)
    - [Third iteration](#third-iteration)
    - [Forth iteration](#forth-iteration)
    - [Validation strategy](#validation-strategy)
  - [Contributing](#contributing)
  - [License](#license)

## <b>Project overview</b>

With varying garbage sorting guidelines across cities, <b>which bin?</b> aims to simplify garbage disposal by identifying trash types and mapping them to the appropriate bins in a user's local area. Users can upload an image of garbage, select their city, and receive accurate disposal guidance.

The application uses a machine learning model to classify garbage into categories such as paper, plastic, and glass. The backend is built with FastAPI, and a front end is available for ease of interaction.

## <b>Features</b>

- <b>Garbage image upload:</b> users can upload an image of their garbage for classification ;
- <b>City-specific sorting rules:</b> based on the user's city selection, the app provides tailored bin recommendations ;
- <b>Multiple model support:</b> users can choose from various machine learning models for classification ;
- <b>Available front end:</b> built with HTML, CSS, and JavaScript for ease of use ;
- <b>Dockerized setup:</b> the application is containerized using Docker for easy deployment.


## <b>Project structure</b>

```plaintext
which-bin/
├── api/
│   └── v1/                 # API version folder (future API versions can go here)
├── app/
│   ├── models/             # Folder for saved ML models
│   ├── services/           # Services used by the app (e.g., ML services)
│   ├── utils/              # Utility functions
│   └── main.py             # FastAPI main app with endpoints and model inference
├── data/
│   ├── augmented_images/   # Folder for storing augmented images for training
│   ├── raw/                # Raw data folder
│   │   └── garbage-dataset/
│   │       ├── battery/
│   │       ├── biological/
│   │       ├── cardboard/
│   │       ├── clothes/
│   │       ├── glass/
│   │       ├── metal/
│   │       ├── paper/
│   │       ├── plastic/
│   │       ├── shoes/
│   │       └── trash/
│   ├── uploaded_images/    # Folder for storing images uploaded by users
│   └── city_bins.json      # JSON file with sorting rules for each city
├── frontend/
│   ├── assets/             # Folder for static assets like images
│   ├── src/                # Source folder for additional front-end code
│   ├── index.html          # Main HTML file for the front end
│   ├── script.js           # JavaScript for front-end interaction
│   └── styles.css          # CSS file for styling
├── mlflows/
│   ├── mlartifacts/        # MLflow artifacts
│   └── mlruns/             # Folder for MLflow runs
├── pipelines/              # Folder for data pipelines and training workflows
├── .gitattributes          # Git attributes for managing file types
├── Dockerfile              # Dockerfile for containerizing the app
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file
├── Roadmap-Notebook.ipynb  # Jupyter notebook for ML model's design
└── README.md               # Project documentation
```
## <b>Installation</b>
### Prerequisites

- Docker
- Python 3.12
- FastAPI

### Setup

- Clone the repository

    git clone https://github.com/yourusername/which-bin.git
    cd which-bin

- Set up docker build the docker image:

    docker build -t whichbin .

- Run the docker container:

    docker run -p 8000:8000 whichbin

- Start the app locally if you're not using docker:

    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

## <b>Usage</b>
Accessing the application

Open a browser and go to http://127.0.0.1:8000 (or http://localhost:8000) to access the front end.

- Choose a city to load specific sorting rules
- Upload a photo of the garbage
- Optionally, choose a model from the drop-down in the top-left corner
- Click the "which bin?" button to see where to dispose of your garbage

## <b>Model training and evaluation</b>
### Dataset Description

The dataset used in this project consists of a collection of images of various types of garbage. The images are categorized into different classes that are: 'battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes' and 'trash'. The dataset is structured into folders, each representing a different class of garbage. The images have varying dimensions, modes, and resolutions. In this dataset there are 19407 images. Corrupted and duplicate images were removed from the dataset. Also, to balance the dataset, minority classes were augmented using transformations such as blur, gaussian filter, zoom, rotation, and horizontal flipping.

### Baseline Model

The baseline model used in this project is a Random forest. The hyperparameter were set on n_estimators=100 and max_depth=10. There were only two preprocessing steps. The first one was to resize the images to 256x256 pixels, maintaining aspect ratio with padding. The second one was to flatten the images into 1D arrays. We've decide to use accuracy for evaluation because our classes are balanced and we want to know how well the model can predict the correct class. The mean accuracy of the base model on a 5 fold cross validation was 48.56%

### First iteration
The first iteration of the model was trained on the dataset with the same feature engineering but with hyperoptimisation tuning using optuna and multiple machine learning family of models such as: the logistic regression, the random forest, the decision tree and the gaussian naive bayes. The best model was a random forest with _estimators=328, max_depth=16, min_samples_split=2. Tis model achieve 55.04% of accuracy.

### Second iteration
The second iteration of the model was trained on the dataset with different feature engineering. The new feature engineering included the addition of the following techniques:
- Histogram of Oriented Gradients (HOG): capturing texture and shape information
- Color histogram: representing the distribution of colors in the image
- Local Binary Pattern (LBP): Ccapturing texture features
We kept using the same hyperparameter tuning as in the first iteration. The best model was a random forest with _estimators=354, max_depth=27, min_samples_split=2,. This model achieved 58.14% of accuracy.

### Third iteration
For the third iteration, we decided to used a pre-trained model for the features extraction task. So we used the ResNet50 model as a feature extractor. We used the same hyperparameter tuning as previously. The best model this time was a logistic regression with C=4.508483888966414 and solver=newton-cg. With that we've achieved 93.07% of accuracy.

### Forth iteration
For the last iteration, we've combine everything we've done in the previous iterations. The feature engineering contains the techniques we've used at the second iteration and the features extracted from the ResNet50 model. We've also used the same hyperparameter tuning as previously. The best model was also a logistic regression C=1.1399428733920607 and solver=newton-cg. With that we've achieved 93.15% of accuracy.

### Validation strategy
The dataset was split into training (80%) and testing (20%) sets, with further splitting of the training set into training and validation subsets (80-20 split). A stratified sampling approach was used to ensure class balance across splits.


## <b>Contributing</b>

Contributions are welcome! If you would like to suggest features, fix issues, or improve documentation:
- Fork the repository
- Create a new branch
- Submit a pull request with detailed comments

## <b>License</b>

This project is licensed under the MIT License. See the LICENSE file for more details.