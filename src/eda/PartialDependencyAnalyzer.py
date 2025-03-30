from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay


class PartialDependenceAnalyzer:
    """Class for creating and visualizing partial dependence plots with XGBoost models"""

    def __init__(self, features, model_params=None, plot_type='average',
                 figsize=(8, 4), nrows=2, ncols=2,
                 threshold=0.5, line_style='--'):
        """
        Initialize the analyzer with configuration parameters

        Args:
            features (list): List of feature names to analyze
            model_params (dict): Parameters for XGBoost classifier
            plot_type (str): Parameters for choose partial dependence vs ice plot
            figsize (tuple): Figure dimensions (width, height)
            nrows (int): Number of rows in subplot grid
            ncols (int): Number of columns in subplot grid
            threshold (float): Decision threshold line position
            line_style (str): Styling for threshold line
        """
        self.features = features
        self.model_params = model_params or {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'n_estimators': 300,
            'max_depth': 4,
            'min_child_weight': 3
        }
        self.plot_type = plot_type
        self.figsize = figsize
        self.nrows = nrows
        self.ncols = ncols
        self.threshold = threshold
        self.line_style = line_style
        self.pipeline = None

    def create_pipeline(self):
        """Build the XGBoost pipeline"""
        xgb = XGBClassifier(**self.model_params)
        self.pipeline = Pipeline([('xgboost', xgb)])
        return self.pipeline

    def fit_and_visualize(self, X, y):
        """
        Train the model and display partial dependence plots
        """
        self.create_pipeline().fit(X, y)
        fig, axes = plt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            figsize=self.figsize
        )
        fig.subplots_adjust(hspace=0.4, wspace=0.2)

        PartialDependenceDisplay.from_estimator(
            self.pipeline,
            X=X,
            kind=self.plot_type,
            subsample=30,
            features=self.features,
            ax=axes
        )

        for ax in axes.flatten():
            ax.axhline(
                self.threshold,
                linestyle=self.line_style,
                color='red'
            )

        plt.show()
