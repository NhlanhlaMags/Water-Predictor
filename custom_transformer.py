from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class WaterFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        original_columns = X.columns.tolist() # Store original columns

        # Log transformations for skewed features (applied to original columns before creating new ones)
        skewed_cols = ['Solids', 'Conductivity', 'Trihalomethanes']
        for col in skewed_cols:
             if col in X.columns:
                X[f'{col}_log'] = np.log1p(X[col])

        # Binary features based on safety thresholds
        # Ensure original columns exist before creating binary features
        if 'ph' in X.columns:
            X['ph_out_of_range'] = ((X['ph'] < 6.5) | (X['ph'] > 8.5)).astype(int)
        if 'Solids' in X.columns:
            X['high_solids'] = (X['Solids'] > 500).astype(int)
        if 'Chloramines' in X.columns:
             # Assuming safe range for Chloramines is 1 to 4 based on common guidelines
            X['chloramine_safe'] = ((X['Chloramines'] >= 1) & (X['Chloramines'] <= 4)).astype(int)
        if 'Sulfate' in X.columns:
            # Assuming high sulfate is > 250 based on some guidelines
            X['sulfate_out_of_range'] = (X['Sulfate'] > 250).astype(int)
        if 'Organic_carbon' in X.columns:
             # Assuming high organic carbon is > 20 (example threshold)
            X['high_organic'] = (X['Organic_carbon'] > 20).astype(int)
        if 'Trihalomethanes' in X.columns:
            # Assuming high Trihalomethanes is > 80 based on some guidelines
            X['trihalo_high'] = (X['Trihalomethanes'] > 80).astype(int)
        if 'Turbidity' in X.columns:
            # Assuming high turbidity is > 5 (example threshold)
            X['turbid'] = (X['Turbidity'] > 5).astype(int)


        # Interaction / ratio features
        # Ensure base columns exist before creating interaction features
        if 'Hardness' in X.columns and 'ph' in X.columns:
            # Add a small constant to the denominator to avoid division by zero
            X['acidity_hardness_ratio'] = X['Hardness'] / (X['ph'] + 1e-8)
        if 'Organic_carbon' in X.columns and 'Trihalomethanes' in X.columns:
             X['organic_contamination_index'] = X['Organic_carbon'] * X['Trihalomethanes']
        if 'Solids' in X.columns and 'Conductivity' in X.columns:
             # Add a small constant to the denominator to avoid division by zero
            X['mineral_index'] = X['Solids'] / (X['Conductivity'] + 1e-8)
        if 'Chloramines' in X.columns and 'ph' in X.columns:
            X['chlorine_impact'] = X['Chloramines'] * X['ph']

        # Handle inf / NaN created by division by zero or other operations
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Impute any NaNs that might have been created
        for col in X.columns:
            if X[col].isnull().any():
                 X[col].fillna(X[col].median(), inplace=True)


        # Polynomial features (nonlinear effects)
        if 'ph' in X.columns:
            X['pH_squared'] = X['ph'] ** 2
        if 'Turbidity' in X.columns:
            X['Turbidity_squared'] = X['Turbidity'] ** 2


        # Water quality stability interactions
        if 'Hardness' in X.columns and 'Conductivity' in X.columns:
             X['hardness_conductivity_interaction'] = X['Hardness'] * X['Conductivity']
        if 'Chloramines' in X.columns and 'Trihalomethanes' in X.columns:
             X['chloramine_trihalomethane_interaction'] = X['Chloramines'] * X['Trihalomethanes']
        if 'ph' in X.columns and 'Turbidity' in X.columns:
             X['ph_turbidity_interaction'] = X['ph'] * X['Turbidity']

        # Mineral composition interactions
        if 'Solids' in X.columns and 'Conductivity' in X.columns:
            X['solid_conductivity_ratio'] = X['Solids'] / (X['Conductivity'] + 1e-8)
        if 'Sulfate' in X.columns and 'Chloramines' in X.columns:
            X['sulfate_chloramine_balance'] = X['Sulfate'] - X['Chloramines']

        # Water treatment efficiency
        if 'Chloramines' in X.columns and 'Organic_carbon' in X.columns:
             X['treatment_efficiency'] = X['Chloramines'] / (X['Organic_carbon'] + 1e-8)

        # Water Safety Score (domain knowledge composite)
        # Ensure all required base columns exist
        required_cols_safety_score = ['ph', 'Turbidity', 'Trihalomethanes', 'Chloramines', 'Organic_carbon']
        if all(col in X.columns for col in required_cols_safety_score):
            X['safety_score'] = (
                (X['ph'].between(6.5, 8.5).astype(int) * 2) +
                (X['Turbidity'] < 5).astype(int) +
                (X['Trihalomethanes'] < 80).astype(int) +
                (X['Chloramines'] > 2).astype(int) +
                (X['Organic_carbon'] < 15).astype(int)
            )
        else:
             # Add a placeholder or default value if not all required columns are present
             X['safety_score'] = 0 # Or some other default

        # Contamination risk index
        # Ensure all required base columns exist
        required_cols_contamination_risk = ['Organic_carbon', 'Turbidity', 'Trihalomethanes', 'ph']
        if all(col in X.columns for col in required_cols_contamination_risk):
            X['contamination_risk'] = (
                X['Organic_carbon'] * 0.3 +
                X['Turbidity'] * 0.25 +
                (X['Trihalomethanes'] > 80).astype(int) * 20 +
                (X['ph'] < 6.5).astype(int) * 15
            )
        else:
            # Add a placeholder or default value if not all required columns are present
            X['contamination_risk'] = 0 # Or some other default


        # Apply log transformation to highly skewed *engineered* features if needed and they exist
        # The original code applied it to some original features and some engineered.
        # This needs careful consideration. If applying to engineered features,
        # do it *after* they are created.
        # Based on the original notebook's highly_skewed_features list, some were original, some engineered.
        # Let's re-evaluate which ones should be log transformed *after* the full set is ready.
        # For simplicity and to match the original notebook's output where possible,
        # I will apply log1p to the list of features identified as highly skewed *in the original notebook*,
        # but do this *after* all features are created.
        # This might re-transform some original columns if they were in the list and also engineered.
        # A more robust approach would be to identify which features to transform *after* the full set is ready.
        # Given the goal is to fix the deployment error and maintain similar behavior,
        # I'll apply log1p to the list from the original notebook's analysis *after* engineering.

        # Re-identify highly skewed features after engineering (optional but more accurate)
        # This requires knowing the state of df after feature engineering in the notebook cells
        # Let's assume the list `highly_skewed_features` from the notebook is the target list for final log transformation
        # This list was: ['Solids_log', 'high_solids', 'chloramine_safe', 'high_organic', 'trihalo_high', 'turbid', 'acidity_hardness_ratio']

        highly_skewed_features_to_log = ['Solids_log', 'high_solids', 'chloramine_safe', 'high_organic', 'trihalo_high', 'turbid', 'acidity_hardness_ratio']

        for col in highly_skewed_features_to_log:
            if col in X.columns:
                 # Add a small constant (1) to handle potential zero values before logging
                 X[col] = np.log1p(X[col])

        # Ensure consistent column order by sorting column names
        # This is crucial for consistent behavior with subsequent pipeline steps
        transformed_columns = X.columns.tolist()
        # Keep original columns in their original order, add new columns sorted
        new_columns = [col for col in transformed_columns if col not in original_columns]
        new_columns.sort()
        final_column_order = original_columns + new_columns
        X = X[final_column_order]


        return X
