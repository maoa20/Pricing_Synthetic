# synthetic_data_pipeline.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from scipy.stats import wasserstein_distance

class DataProcessor:
    def __init__(self, categorical_features=None, numerical_features=None):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.preprocessor = None

    def fit_transform(self, df, target_column):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), self.categorical_features),
                ('num', StandardScaler(), self.numerical_features)
            ], remainder='passthrough')

        X = df[self.categorical_features + self.numerical_features]
        y = df[target_column]

        X_transformed = self.preprocessor.fit_transform(X)
        return X_transformed, y

    def transform(self, df):
        X = df[self.categorical_features + self.numerical_features]
        X_transformed = self.preprocessor.transform(X)
        return X_transformed


class SyntheticDataModel:
    def __init__(self, model=None, n_splits=5, random_state=42):
        self.model = model if model else RandomForestRegressor()
        self.n_splits = n_splits
        self.random_state = random_state
        self.synthetic_data = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def generate(self, X_new):
        return self.model.predict(X_new)

    def kfold_validate(self, X, y_real):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_losses = []
        synthetic_data = []

        for train_index, test_index in kf.split(X):
            X_fold, y_fold_real = X[train_index], y_real.iloc[train_index]
            y_fold_synthetic = self.generate(X_fold)
            synthetic_data.append(y_fold_synthetic)

            fold_loss = wasserstein_distance(y_fold_real, y_fold_synthetic)
            fold_losses.append(fold_loss)

        self.synthetic_data = np.concatenate(synthetic_data, axis=0)
        return np.mean(fold_losses)

    def run(self, df, weeks_column, target_column, categorical_features, numerical_features, max_weeks=2):
        processor = DataProcessor(categorical_features, numerical_features)
        results = []

        weeks = sorted(df[weeks_column].unique())
        synthetic_data_dict = {}

        for i in range(min(len(weeks) - 1, max_weeks - 1)):  # Limitar a `max_weeks`
            train_week = weeks[i]
            validate_week = weeks[i + 1]

            train_data = df[df[weeks_column] == train_week]
            validate_data = df[df[weeks_column] == validate_week]

            # Preprocesamiento dos dados
            X_train, y_train = processor.fit_transform(train_data, target_column=target_column)
            X_validate, y_validate = processor.transform(validate_data), validate_data[target_column]

            # Treinamento e validação
            self.fit(X_train, y_train)
            average_loss = self.kfold_validate(X_validate, y_validate)

            # Salvando so dados
            synthetic_data_dict[validate_week] = self.synthetic_data

            print(f'Week {i} {train_week} -> {validate_week}: Average Wasserstein Loss = {average_loss:.4f}')
            results.append(average_loss)

        return results, synthetic_data_dict


def predict_for_specific_week(model, df, week, weeks_column, categorical_features, numerical_features):
    processor = DataProcessor(categorical_features, numerical_features)
    week_data = df[df[weeks_column] == week]

    X_week, _ = processor.fit_transform(week_data, target_column='Quantity')  

    synthetic_quantities = model.generate(X_week)

    synthetic_data_df = week_data[[weeks_column, 'ModelID', 'ItemColorId']].copy()
    synthetic_data_df['Quantity_Synthetic'] = synthetic_quantities

    return synthetic_data_df


