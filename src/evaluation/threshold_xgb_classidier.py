from xgboost import XGBClassifier


class ThresholdXGBClassifier(XGBClassifier):
    def __init__(self, xgb, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.xgb = xgb


    def predict(self, X, *args, **kwargs):
        """
            Predict with `threshold` applied to predicted class probabilities.
        """
        proba = self.xgb.predict_proba(X, *args, **kwargs)
        return (proba[:, 1] > self.threshold).astype(int)
