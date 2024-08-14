# Blood Sample Classifier

## Overview

This project involves the development of a binary classifier designed to distinguish between human and animal blood samples. The classifier is trained to identify blood samples from humans and four different animal species: cat, cow, dog, and goat. The classification is based on the chemical structure of the blood, which was analyzed using Fourier-Transform Infrared Spectroscopy (FTIR). 

## Objectives

- **Collect Blood Samples**: Gather blood samples from humans and four animal species.
- **Analyze Samples**: Use FTIR spectroscopy to analyze the chemical structure of the blood samples.
- **Preprocess Data**: Clean and prepare the dataset using Python and Orange.
- **Build Classifier**: Implement a binary classifier using Partial Least Squares Discriminant Analysis (PLS-DA) to discriminate between human and animal blood.
- **Achieve Accuracy**: The classifier achieved approximately 99% accuracy during training and testing.

## FTIR Spectroscopy

FTIR (Fourier-Transform Infrared) Spectroscopy is an analytical technique used to understand the chemical structure of both organic and inorganic materials. The method involves shining infrared light on a sample and measuring the absorption of this light to determine the chemical properties of the sample.

- **Technique**: Measures how much infrared light is absorbed by the sample.
- **Purpose**: Obtains an infrared spectrum to analyze the chemical composition.

## PLS-DA (Partial Least Squares Discriminant Analysis)

PLS-DA is a statistical method used for classification tasks, especially with high-dimensional data. It is a supervised dimensionality reduction technique that helps in classifying data based on its features.

- **Purpose**: Classify data into predefined classes.
- **Usage**: Effective for high-dimensional datasets.

## Dataset

- **Source**: Blood samples collected from humans and animals (cat, cow, dog, goat).
- **Analysis**: FTIR spectroscopy to obtain spectral data.
- **Preprocessing**: Data cleaned and organized using Python and Orange.

## Implementation

The classifier was implemented using the PLS-DA algorithm, which provided high accuracy in distinguishing between human and animal blood samples.
