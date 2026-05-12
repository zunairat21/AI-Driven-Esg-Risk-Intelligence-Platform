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
<<<<<<< HEAD
    
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

        
=======

    def build_explainer(self, X_transformed):
        """
        Create SHAP explainer (TreeExplainer for tree models)
        """
        self.explainer = shap.Explainer(self.model, X_transformed)
        return self.explainer

    def get_shap_values(self, X_transformed):
        """
        Compute SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not built. Call build_explainer first.")
        return self.explainer(X_transformed)

    def plot_summary(self, shap_values, X_transformed):
        """
        Global feature importance
        """
        shap.summary_plot(shap_values, X_transformed)

    def plot_force(self, shap_values, X_transformed, index=0):
        """
        Local explanation for a single prediction
        """
        shap.force_plot(
            self.explainer.expected_value,
            shap_values[index].values,
            X_transformed[index]
        )

          
  
   



                         

  
  
    
  

>>>>>>> 12a2ba96de2e51100dde7126d21e0eea61e8e6b4
