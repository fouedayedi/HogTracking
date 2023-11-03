# HOG Feature Extraction and SVM Classification

This repository contains the implementation and evaluation of object detection using Histogram of Oriented Gradients (HOG) features and Support Vector Machine (SVM) classification. 

## Project Structure

```plaintext
.
├── data                 # Directory containing the dataset
├── model                
│   └── classHOG.p       # Pre-trained model
├── pickle              
├── utile                # Utility functions and scripts
│   ├── __init__.py     
│   ├── functions.py     # Utility functions
│   └── visuHOG.py       # Visualization for HOG
├── .gitignore
├── HOG_ATELIER1.py      # SVM training and results
├── README.md
├── main.py              # Main script to run the project
└── requirements.txt     # List of dependencies
```
## Dataset

The dataset used for this project is the INRIA Person Dataset. It is a popular dataset used for the task of person detection.

- [Download the dataset here](https://drive.google.com/u/0/uc?id=14GD_pBpBsprPiZlkmtXN_y5K72To16if&export=download)
- [More about the INRIA dataset](https://paperswithcode.com/dataset/inria-person)

## Key Reference

For those interested in diving deeper into the topic, the following paper provides an excellent analysis of why linear SVMs trained on HOG features perform exceptionally well:

- "Why do linear SVMs trained on HOG features perform so well?" [arXiv:1406.2419](https://arxiv.org/abs/1406.2419)

## Setup and Installation

1. Clone this repository:

2. Install the required dependencies:

3. Download and extract the INRIA dataset.

4. Run the main script:
python main.py




