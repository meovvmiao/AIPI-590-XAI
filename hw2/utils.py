import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

class ModelAssumptionChecker:
    """
    A comprehensive class to check assumptions for Linear Regression, 
    Logistic Regression, and GAM models through EDA
    """
    
    def __init__(self, data, target_column, feature_columns=None):
        """
        Initialize the assumption checker
        
        Parameters:
        data: pandas DataFrame
        target_column: str, name of target variable
        feature_columns: list, names of feature variables (if None, uses all except target)
        """
        self.data = data.copy()
        self.target = target_column
        self.features = feature_columns if feature_columns else [col for col in data.columns if col != target_column]
        self.y = self.data[self.target]
        self.X = self.data[self.features]
        
    def linear_regression_assumptions(self):
        """Check assumptions for linear regression model"""
        print("="*60)
        print("LINEAR REGRESSION ASSUMPTION CHECKS")
        print("="*60)
        
        # Fit linear model
        lr_model = LinearRegression()
        lr_model.fit(self.X, self.y)
        y_pred = lr_model.predict(self.X)
        residuals = self.y - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Linear Regression Assumption Checks', fontsize=16, fontweight='bold')
        
        # 1. Linearity Check - Scatter plots of features vs target
        ax = axes[0, 0]
        if len(self.features) == 1:
            ax.scatter(self.X.iloc[:, 0], self.y, alpha=0.6, color='blue')
            ax.plot(self.X.iloc[:, 0], y_pred, color='red', linewidth=2)
            ax.set_xlabel(self.features[0])
            ax.set_ylabel(self.target)
            ax.set_title('Linearity Check')
        else:
            # For multiple features, show correlation matrix
            corr_matrix = pd.concat([self.X, self.y], axis=1).corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation Matrix')
        
        # 2. Residuals vs Fitted (Homoscedasticity)
        ax = axes[0, 1]
        ax.scatter(y_pred, residuals, alpha=0.6, color='blue')
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Fitted\n(Homoscedasticity Check)')
        
        # 3. Normal Q-Q plot of residuals
        ax = axes[0, 2]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals\n(Normality Check)')
        
        # 4. Histogram of residuals
        ax = axes[1, 0]
        ax.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        # Overlay normal curve
        mu, sigma = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Residuals')
        ax.legend()
        
        # 5. Scale-Location plot
        ax = axes[1, 1]
        standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
        ax.scatter(y_pred, standardized_residuals, alpha=0.6, color='green')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('√|Standardized Residuals|')
        ax.set_title('Scale-Location Plot')
        
        # 6. Leverage and Cook's distance (if sample size permits)
        ax = axes[1, 2]
        if len(self.data) > len(self.features) + 1:
            # Calculate leverage
            X_with_intercept = np.column_stack([np.ones(len(self.X)), self.X])
            hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
            leverage = np.diag(hat_matrix)
            ax.scatter(range(len(leverage)), leverage, alpha=0.6, color='orange')
            ax.axhline(y=2*(len(self.features)+1)/len(self.data), color='red', linestyle='--', 
                      label='2(p+1)/n threshold')
            ax.set_xlabel('Observation Index')
            ax.set_ylabel('Leverage')
            ax.set_title('Leverage Plot')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Not enough data\nfor leverage calculation', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Leverage Plot')
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        print("\nSTATISTICAL TESTS:")
        print("-" * 30)
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"Shapiro-Wilk normality test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        
        ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        print(f"Kolmogorov-Smirnov normality test: D={ks_stat:.4f}, p={ks_p:.4f}")
        
        # Homoscedasticity test (Breusch-Pagan)
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, self.X)
            print(f"Breusch-Pagan homoscedasticity test: LM={bp_stat:.4f}, p={bp_p:.4f}")
        except:
            print("Breusch-Pagan test could not be performed")
        
        # Autocorrelation test (Durbin-Watson)
        dw_stat = durbin_watson(residuals)
        print(f"Durbin-Watson autocorrelation test: DW={dw_stat:.4f}")
        print("  (Values close to 2 indicate no autocorrelation)")
        
        # Model fit
        r2 = r2_score(self.y, y_pred)
        print(f"\nModel R-squared: {r2:.4f}")
        
    def logistic_regression_assumptions(self):
        """Check assumptions for logistic regression model"""
        print("\n" + "="*60)
        print("LOGISTIC REGRESSION ASSUMPTION CHECKS")
        print("="*60)
        
        # Check if target is binary
        unique_values = self.y.unique()
        if len(unique_values) != 2:
            print(f"Warning: Target variable has {len(unique_values)} unique values.")
            print("Logistic regression typically requires binary target variable.")
            print(f"Unique values: {unique_values}")
            return
        
        # Convert target to binary if needed
        y_binary = (self.y == unique_values[1]).astype(int)
        
        # Fit logistic model
        lr_model = LogisticRegression(max_iter=1000)
        lr_model.fit(self.X, y_binary)
        
        # Get probabilities
        probabilities = lr_model.predict_proba(self.X)[:, 1]
        log_odds = np.log(probabilities / (1 - probabilities + 1e-10))  # Add small value to avoid division by zero
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Logistic Regression Assumption Checks', fontsize=16, fontweight='bold')
        
        # 1. Linearity of log-odds (Box-Tidwell test approximation)
        ax = axes[0, 0]
        if len(self.features) == 1:
            ax.scatter(self.X.iloc[:, 0], log_odds, alpha=0.6, color='blue')
            ax.set_xlabel(self.features[0])
            ax.set_ylabel('Log-odds')
            ax.set_title('Linearity of Log-odds')
        else:
            # For multiple features, show first feature
            ax.scatter(self.X.iloc[:, 0], log_odds, alpha=0.6, color='blue')
            ax.set_xlabel(f'{self.features[0]} (first feature)')
            ax.set_ylabel('Log-odds')
            ax.set_title('Linearity of Log-odds\n(First Feature)')
        
        # 2. Distribution of target variable
        ax = axes[0, 1]
        target_counts = pd.Series(y_binary).value_counts()
        ax.bar(['Class 0', 'Class 1'], target_counts.values, color=['lightcoral', 'skyblue'])
        ax.set_ylabel('Count')
        ax.set_title('Target Variable Distribution')
        for i, v in enumerate(target_counts.values):
            ax.text(i, v + max(target_counts.values)*0.01, str(v), ha='center')
        
        # 3. Predicted probabilities distribution
        ax = axes[1, 0]
        ax.hist(probabilities, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Predicted Probabilities')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predicted Probabilities')
        ax.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
        ax.legend()
        
        # 4. Multicollinearity check (VIF approximation using correlation)
        ax = axes[1, 1]
        if len(self.features) > 1:
            corr_matrix = self.X.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlation\n(Multicollinearity Check)')
        else:
            ax.text(0.5, 0.5, 'Single feature:\nNo multicollinearity', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Multicollinearity Check')
        
        plt.tight_layout()
        plt.show()
        
        # Statistical information
        print("\nSTATISTICAL INFORMATION:")
        print("-" * 30)
        print(f"Target variable balance: {target_counts[0]} vs {target_counts[1]}")
        print(f"Proportion: {target_counts[0]/len(y_binary):.3f} vs {target_counts[1]/len(y_binary):.3f}")
        
        # Check for separation
        prob_range = probabilities.max() - probabilities.min()
        print(f"Predicted probability range: {probabilities.min():.4f} to {probabilities.max():.4f}")
        if prob_range < 0.1:
            print("Warning: Small probability range may indicate separation issues")
        
        # Feature correlation check
        if len(self.features) > 1:
            max_corr = np.abs(self.X.corr().values[np.triu_indices_from(self.X.corr().values, k=1)]).max()
            print(f"Maximum feature correlation: {max_corr:.4f}")
            if max_corr > 0.8:
                print("Warning: High correlation detected - potential multicollinearity")
    
    def gam_assumptions(self):
        """Check assumptions and suitability for GAM models"""
        print("\n" + "="*60)
        print("GAM (GENERALIZED ADDITIVE MODEL) SUITABILITY CHECKS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GAM Suitability Assessment', fontsize=16, fontweight='bold')
        
        # 1. Non-linear relationships check
        for i, feature in enumerate(self.features[:6]):  # Show up to 6 features
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                ax = axes[row, col]
                
                # Scatter plot with LOWESS smoothing
                ax.scatter(self.X[feature], self.y, alpha=0.6, color='lightblue')
                
                # Add LOWESS smooth line
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    smoothed = lowess(self.y, self.X[feature], frac=0.3)
                    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2, label='LOWESS')
                    
                    # Add linear fit for comparison
                    z = np.polyfit(self.X[feature], self.y, 1)
                    p = np.poly1d(z)
                    ax.plot(self.X[feature], p(self.X[feature]), 'g--', alpha=0.8, label='Linear')
                    ax.legend()
                except:
                    # If LOWESS not available, just show scatter
                    pass
                
                ax.set_xlabel(feature)
                ax.set_ylabel(self.target)
                ax.set_title(f'Non-linearity: {feature}')
        
        # Fill remaining subplots if fewer than 6 features
        for i in range(len(self.features), 6):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation and interaction analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature correlation heatmap
        ax = axes[0]
        corr_with_target = pd.concat([self.X, self.y], axis=1).corr()
        sns.heatmap(corr_with_target, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix\n(Features + Target)')
        
        # Distribution of target variable
        ax = axes[1]
        ax.hist(self.y, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel(self.target)
        ax.set_ylabel('Frequency')
        ax.set_title('Target Variable Distribution')
        
        # Add normal curve if continuous
        if len(self.y.unique()) > 10:  # Assume continuous if many unique values
            mu, sigma = stats.norm.fit(self.y)
            x = np.linspace(self.y.min(), self.y.max(), 100)
            ax2 = ax.twinx()
            ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Density', color='red')
        
        plt.tight_layout()
        plt.show()
        
        # GAM suitability assessment
        print("\nGAM SUITABILITY ASSESSMENT:")
        print("-" * 30)
        
        # Check for non-linear patterns
        print("Non-linearity indicators:")
        for feature in self.features:
            # Calculate correlation coefficient
            linear_corr = np.corrcoef(self.X[feature], self.y)[0, 1]
            
            # Simple non-linearity test using polynomial fit
            try:
                # Fit linear and quadratic models
                linear_fit = np.polyfit(self.X[feature], self.y, 1)
                quad_fit = np.polyfit(self.X[feature], self.y, 2)
                
                linear_pred = np.polyval(linear_fit, self.X[feature])
                quad_pred = np.polyval(quad_fit, self.X[feature])
                
                linear_r2 = r2_score(self.y, linear_pred)
                quad_r2 = r2_score(self.y, quad_pred)
                
                improvement = quad_r2 - linear_r2
                print(f"  {feature}: Linear R² = {linear_r2:.4f}, Quadratic R² = {quad_r2:.4f}")
                print(f"    Improvement: {improvement:.4f} {'(suggests non-linearity)' if improvement > 0.05 else ''}")
                
            except:
                print(f"  {feature}: Could not compute non-linearity test")
        
        # Target variable characteristics
        print(f"\nTarget variable characteristics:")
        print(f"  Type: {'Continuous' if len(self.y.unique()) > 10 else 'Discrete/Categorical'}")
        print(f"  Range: {self.y.min():.4f} to {self.y.max():.4f}")
        print(f"  Mean: {self.y.mean():.4f}, Std: {self.y.std():.4f}")
        
        # Sample size assessment
        print(f"\nSample size assessment:")
        print(f"  Total observations: {len(self.data)}")
        print(f"  Features: {len(self.features)}")
        print(f"  Observations per feature: {len(self.data)/len(self.features):.1f}")
        if len(self.data)/len(self.features) < 10:
            print("  Warning: Low observations per feature ratio for GAM")
    
    def run_all_checks(self):
        """Run all assumption checks"""
        print("COMPREHENSIVE MODEL ASSUMPTION ANALYSIS")
        print("=" * 60)
        print(f"Dataset shape: {self.data.shape}")
        print(f"Target variable: {self.target}")
        print(f"Features: {self.features}")
        print()
        
        # Basic data info
        print("DATA OVERVIEW:")
        print("-" * 20)
        print(self.data.describe())
        print()
        
        # Check for missing values
        missing_data = self.data.isnull().sum()
        if missing_data.any():
            print("MISSING VALUES:")
            print("-" * 20)
            print(missing_data[missing_data > 0])
            print()
        
        # Run specific model checks
        self.linear_regression_assumptions()
        self.logistic_regression_assumptions()
        self.gam_assumptions()

# Example usage function
def example_usage():
    """Example of how to use the ModelAssumptionChecker"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    # Create features with different relationships
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    
    # Linear target
    y_linear = 2*x1 + 1.5*x2 - 0.8*x3 + np.random.normal(0, 0.5, n_samples)
    
    # Non-linear target (for GAM demonstration)
    y_nonlinear = 2*x1 + 1.5*np.sin(x2*2) - 0.8*x3**2 + np.random.normal(0, 0.5, n_samples)
    
    # Binary target (for logistic regression)
    probabilities = 1 / (1 + np.exp(-(2*x1 + 1.5*x2 - 0.8*x3)))
    y_binary = np.random.binomial(1, probabilities, n_samples)
    
    # Create DataFrames
    linear_data = pd.DataFrame({
        'feature1': x1,
        'feature2': x2, 
        'feature3': x3,
        'target': y_linear
    })
    
    binary_data = pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'feature3': x3,
        'target': y_binary
    })
    
    nonlinear_data = pd.DataFrame({
        'feature1': x1,
        'feature2': x2,
        'feature3': x3,
        'target': y_nonlinear
    })
    
    print("Example 1: Linear Regression Data")
    checker1 = ModelAssumptionChecker(linear_data, 'target')
    checker1.linear_regression_assumptions()
    
    print("\n\nExample 2: Logistic Regression Data")
    checker2 = ModelAssumptionChecker(binary_data, 'target')
    checker2.logistic_regression_assumptions()
    
    print("\n\nExample 3: Non-linear Data (GAM suitable)")
    checker3 = ModelAssumptionChecker(nonlinear_data, 'target')
    checker3.gam_assumptions()

