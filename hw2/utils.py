import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For Linear/Logistic Assumptions
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For GAMs
from pygam import LinearGAM, LogisticGAM, s, f

# Set plot style
sns.set_style("whitegrid")

class ModelAssumptionEDA:
    """
    A class to perform Exploratory Data Analysis (EDA) to check assumptions
    for Linear Models (Linear/Logistic Regression) and GAMs.

    Args:
        data (pd.DataFrame): The input dataframe.
        target_variable (str): The name of the target variable column.
        model_type (str): Type of model, either 'linear' for regression or
                          'logistic' for classification.
    """
    def __init__(self, data, target_variable, model_type):
        if model_type not in ['linear', 'logistic']:
            raise ValueError("model_type must be 'linear' or 'logistic'")
        if target_variable not in data.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data.")

        self.data = data.copy()
        self.target = target_variable
        self.model_type = model_type
        self.features = [col for col in self.data.columns if col != self.target]

        # Identify feature types
        self.numeric_features = self.data[self.features].select_dtypes(include=np.number).columns.tolist()
        self.text_features = self.data[self.features].select_dtypes(include=['object', 'category']).columns.tolist()

        print("--- EDA Setup ---")
        print(f"Model Type: {self.model_type.capitalize()}")
        print(f"Target Variable: {self.target}")
        print(f"Identified {len(self.numeric_features)} numeric features: {self.numeric_features}")
        print(f"Identified {len(self.text_features)} text/categorical features: {self.text_features}\n")

        # Prepare data for modeling
        self._prepare_data()

    def _prepare_data(self):
        """Preprocesses the data by imputing and encoding features."""
        # Define preprocessing pipelines for numeric and text features
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        text_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        # Create a preprocessor object using ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, self.numeric_features),
                ('cat', text_pipeline, self.text_features)
            ],
            remainder='passthrough'
        )

        # Fit and transform the data
        self.X = self.data[self.features]
        self.y = self.data[self.target]
        self.X_processed = self.preprocessor.fit_transform(self.X)
        
        # Get feature names after one-hot encoding
        try:
            ohe_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(self.text_features)
            self.processed_feature_names = self.numeric_features + list(ohe_feature_names)
        except AttributeError: # For older sklearn versions
             ohe_feature_names = self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_(self.text_features)
             self.processed_feature_names = self.numeric_features + list(ohe_feature_names)


    def check_linear_model_assumptions(self):
        """
        Runs EDA checks for Linear or Logistic Regression assumptions.
        1. Linearity
        2. Multicollinearity
        3. For Linear Regression Only:
           - Normality of Residuals
           - Homoscedasticity (Constant Variance of Residuals)
        """
        print("\n--- Checking Linear Model Assumptions ---")
        self._check_linearity()
        self._check_multicollinearity()
        
        if self.model_type == 'linear':
            self._check_residuals()

    def _check_linearity(self):
        """
        Assesses the linearity assumption.
        - For Linear Regression: Scatter plot of each feature vs. target.
        - For Logistic Regression: Box plots of each feature vs. target classes.
        """
        print("\n1. Checking Linearity Assumption...")
        num_features = len(self.numeric_features)
        if num_features == 0:
            print("No numeric features to check for linearity.")
            return
            
        plt.figure(figsize=(5 * min(num_features, 3), 5 * ((num_features - 1) // 3 + 1)))
        
        for i, col in enumerate(self.numeric_features, 1):
            ax = plt.subplot((num_features - 1) // 3 + 1, min(num_features, 3), i)
            if self.model_type == 'linear':
                sns.regplot(x=col, y=self.target, data=self.data, scatter_kws={'alpha':0.5}, line_kws={'color': 'red'}, ax=ax)
                ax.set_title(f'{col} vs. {self.target}')
            else: # logistic
                sns.boxplot(x=self.target, y=col, data=self.data, ax=ax)
                ax.set_title(f'Distribution of {col} by {self.target}')
        
        plt.suptitle("Linearity Assumption Check", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def _check_multicollinearity(self):
        """
        Checks for multicollinearity using a correlation heatmap and VIF.
        """
        print("\n2. Checking for Multicollinearity...")
        
        # Correlation Heatmap
        corr = self.data[self.numeric_features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap of Numeric Features')
        plt.show()
        
        # VIF (Variance Inflation Factor)
        X_numeric_processed = pd.DataFrame(
            self.preprocessor.named_transformers_['num'].fit_transform(self.data[self.numeric_features]),
            columns=self.numeric_features
        )
        # Add a constant for VIF calculation
        X_vif = sm.add_constant(X_numeric_processed)
        
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        
        print("Variance Inflation Factor (VIF):")
        print(vif_data[vif_data['feature'] != 'const'])
        print("\n(Note: VIF > 5 is a concern, VIF > 10 is a strong concern)\n")

    def _check_residuals(self):
        """
        Checks residual assumptions for Linear Regression:
        - Homoscedasticity (Residuals vs. Fitted plot)
        - Normality of Residuals (Q-Q Plot and Histogram)
        """
        print("\n3. Checking Residuals (for Linear Regression only)...")
        # Fit a simple OLS model to get residuals
        X_with_const = sm.add_constant(self.X_processed)
        model = sm.OLS(self.y, X_with_const).fit()
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Residuals vs. Fitted Plot (Homoscedasticity)
        sns.scatterplot(x=fitted_values, y=residuals, ax=axes[0], alpha=0.6)
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_xlabel('Fitted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs. Fitted Plot')
        
        # Q-Q Plot (Normality)
        sm.qqplot(residuals, line='45', fit=True, ax=axes[1])
        axes[1].set_title('Q-Q Plot of Residuals')
        
        # Histogram of Residuals (Normality)
        sns.histplot(residuals, kde=True, ax=axes[2])
        axes[2].set_title('Histogram of Residuals')
        
        plt.suptitle("Residual Analysis", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def check_gam_assumptions(self):
        """
        Runs EDA for GAMs to identify non-linear relationships.
        This is done by fitting a preliminary GAM and plotting the partial dependencies.
        """
        print("\n--- Checking for Non-Linear Relationships (for GAMs) ---")
        
        if len(self.numeric_features) == 0:
            print("No numeric features to analyze for non-linear relationships.")
            return

        # Build GAM terms: splines for numeric, factors for text
        gam_terms = None
        for i, feature in enumerate(self.numeric_features):
            term = s(i) # s() creates a spline term
            gam_terms = term if gam_terms is None else gam_terms + term
        
        # Add factor terms for categorical features
        # Note: GAMs handle one-hot encoded features gracefully with f()
        start_idx = len(self.numeric_features)
        for cat_col in self.text_features:
            num_levels = len(self.data[cat_col].unique()) -1 # -1 because we dropped one level
            if num_levels > 0:
              indices = list(range(start_idx, start_idx + num_levels))
              term = f(indices[0]) # f() creates a factor term for categorical features
              for j in range(1, len(indices)):
                  term += f(indices[j])
              gam_terms += term
              start_idx += num_levels

        # Fit a preliminary GAM
        if self.model_type == 'linear':
            gam = LinearGAM(gam_terms).fit(self.X_processed, self.y)
        else: # logistic
            gam = LogisticGAM(gam_terms).fit(self.X_processed, self.y)

        # Plot partial dependencies
        n_features_to_plot = len(self.numeric_features)
        fig, axes = plt.subplots(1, n_features_to_plot, figsize=(7 * n_features_to_plot, 6))
        
        if n_features_to_plot == 1: # Matplotlib returns a single axes object if only one plot
            axes = [axes]

        for i, ax in enumerate(axes):
            pdp, confi = gam.partial_dependence(term=i, X=self.X_processed, width=0.95)
            ax.plot(self.X_processed[:, i], pdp)
            ax.plot(self.X_processed[:, i], confi, c='r', ls='--')
            ax.set_title(f'Partial Dependence for {self.numeric_features[i]}')
            ax.set_xlabel(f'{self.numeric_features[i]} (Standardized)')
            ax.set_ylabel('Partial Dependence')

        plt.suptitle("GAM Partial Dependence Plots", fontsize=18, y=1.03)
        plt.tight_layout()
        plt.show()

