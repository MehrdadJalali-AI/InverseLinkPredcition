# Inverse Link Prediction in Metal-Organic Frameworks (MOFs) Graphs


[![Watch the video](ILP2.jpg)](https://youtu.be/W8EMoVmhp_A)


## Overview
This repository is developed under the guidance of Dr. Mehrdad Jalali and focuses on addressing computational challenges in characterizing Metal-Organic Frameworks (MOFs) using an innovative graph sparsification method. This method, termed Inverse Link Prediction (ILP), leverages Graph Convolutional Networks (GCN) for enhanced analysis.

## Concept
The Inverse Link Prediction (ILP) technique introduced in this project is tailored to the structural dynamics of MOFs. It refines MOF-based networks by selectively pruning less critical connections, preserving essential structural integrity. This approach significantly enhances the manageability and efficiency of networks, facilitating more effective computational analyses.

## Theoretical Background
The foundation of ILP in MOFs is predicated on the differential impact of links within the network. By strategically removing less critical links, we reduce network complexity without undermining the capacity for accurate predictions of crucial properties such as gas adsorption capacities, demonstrated here with metrics like the Pore Limiting Diameter (PLD).

## Repository Contents
- **Code/**: Python scripts and modules implementing the ILP through GCN on MOF data.
- **Data/**: Datasets representing MOF structures and their respective properties.
- **Docs/**: Documentation detailing the algorithms, their applications, and theoretical underpinnings.

## Evaluation
The effectiveness of our sparsification method is assessed through metrics such as network parameter centralities and machine learning predictions. Our method has proven highly effective in predicting essential MOF properties, enhancing both theoretical understanding and practical applications in material science.

## How to Use
To utilize the scripts, please ensure your environment is prepared with the required Python version and libraries as listed in `requirements.txt`. Detailed setup instructions can be found in the installation guide.

## Contribution
We welcome contributions to enhance and expand this project. Please feel free to fork the repository, submit pull requests, or open issues to propose changes or discuss enhancements.

## License
This project is made available under the MIT License - see the [LICENSE](LICENSE.md) file for details.
