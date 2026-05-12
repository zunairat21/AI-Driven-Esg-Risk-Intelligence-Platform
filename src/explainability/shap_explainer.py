import shap
import joblib

class ShapExplainer:
    def __init__(self, pipeline_path):
        
        """
        Load full sklearn pipeline
        """
        self.pipeline = joblib.load(pipeline_path)
        
        #Extract components
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.model = self.pipeline.named_steps["model"]
        
        self.explainer = None

    def preprocess_data(self, X):

        """
        Apply preprocessing pipeline
        """
        return self.preprocessor.transform(X)
    
    def get_features_names(self):
        """
        Get transformed feature names out
        """
        return self.preprocessor.get_feature_names_out()
    
    def build_explainer(self, X_transformed):

        """
        Build SHAP Explainer
        """
        self.explainer = shap.Explainer(self.model, X_transformed)

    def compute_shap_values(self, X_transformed):
        """
        Generate shap values
        """
        return self.explainer(X_transformed)
     
    def summary_plot(self, shap_values, X_transformed):

        feature_names=self.get_features_names()
        """
        Global Feature Importance
        """
        shap.summary_plot(
            shap_values, 
            X_transformed,
            feature_names = feature_names
            )
        
        import matplotlib.pyplot as plt
        plt.savefig("assets/shap_summary_plot.png", bbox_inches="tight")
        
