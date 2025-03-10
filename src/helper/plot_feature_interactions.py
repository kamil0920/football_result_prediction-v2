import matplotlib.pyplot as plt
import numpy as np


def plot_feature_interaction(feature_names, shap_interactions, feature1, feature2):
    """
    Plots the interaction between two specific features using SHAP interaction values.

    Parameters:
        feature_names (list): List of feature names.
        shap_interactions (np.array): SHAP interaction values matrix.
        feature1 (str): Name of the first feature.
        feature2 (str): Name of the second feature.
    """
    idx1 = feature_names.index(feature1)
    idx2 = feature_names.index(feature2)

    interaction_values = shap_interactions[:, idx1, idx2]

    plt.scatter(range(len(interaction_values)), interaction_values, alpha=0.7)
    plt.title(f"Interaction Between {feature1} and {feature2}")
    plt.xlabel("Sample Index")
    plt.ylabel("Interaction Value")
    plt.grid(True)
    plt.show()


from sklearn.inspection import PartialDependenceDisplay


def plot_pdp(model, X, features):
    """
    Plots a Partial Dependence Plot (PDP) for the given features.

    Parameters:
        model: Trained machine learning model.
        X (pd.DataFrame): Input dataset.
        features (list): List of two features to analyze interactions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)

    plt.title(f"Partial Dependence Plot for {features[0]} and {features[1]}")
    plt.show()


import seaborn as sns


def plot_scatter_interaction(data, x_feature, y_feature, hue_feature):
    """
    Creates a scatterplot to visualize interactions between two features with color coding by a third variable.

    Parameters:
        data (pd.DataFrame): Input dataset.
        x_feature (str): Feature name for x-axis.
        y_feature (str): Feature name for y-axis.
        hue_feature (str): Feature name for color coding.
    """
    sns.scatterplot(data=data, x=x_feature, y=y_feature, hue=hue_feature, alpha=0.7)
    plt.title(f"Interaction Between {x_feature} and {y_feature} Colored by {hue_feature}")
    plt.show()
