# Inverse Link Prediction in Social Graphs

![ILP.jpg](./ILP.jpg)

## Overview
This repository is dedicated to the study and application of an innovative graph sparsification method using Inverse Link Prediction (ILP) through Graph Convolutional Networks (GCN). Developed by Dr. Mehrdad Jalali, this work builds upon our previous framework, MOFGalaxyNet, which models social networks using Metal-Organic Frameworks (MOFs) data.

## Concept
The ILP method introduced here focuses on refining networks by selectively pruning less critical connections while preserving the essential structural integrity needed for effective computational analysis. The Graph Convolutional Networks (GCN) aspect of ILP specifically targets and evaluates the strength and relevance of each link within the network by leveraging node features and their connectivity patterns. This strategic removal of links minimizes impact on the network's functional characteristics, maintaining vital connections and enhancing the network's manageability and efficiency.

## Theoretical Background
The core theory behind ILP is based on the premise that not all connections in a network contribute equally to its functional integrity. By identifying and removing less impactful links, we can reduce the complexity of the network without sacrificing its ability to accurately predict essential properties. This is particularly crucial in applications like gas adsorption where properties such as Pore Limiting Diameter (PLD) are vital.

## Repository Contents
- **Code/**: All Python scripts and modules used to implement the ILP through GCN.
- **Data/**: Sample datasets used for training and testing the models.
- **Docs/**: Additional documentation related to the algorithms and their implementation.

## Evaluation
We have evaluated the effectiveness of our sparsification method using various metrics, including network parameter centralities and machine learning predictions of properties like Pore Limiting Diameter (PLD). Case studies have demonstrated high accuracy in predicting key functionalities, confirming the practical applications of our method in material science.

## How to Use
To run the scripts in this repository, ensure that you have Python installed along with the necessary libraries listed in `requirements.txt`. Follow the instructions in the installation guide to set up your environment.

## Contribution
Contributions to this project are welcome. Please submit a pull request or open an issue to discuss potential changes or additions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
