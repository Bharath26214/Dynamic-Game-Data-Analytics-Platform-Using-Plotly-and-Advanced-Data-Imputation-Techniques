import xgboost
import optuna

class Imputer:
    """
    A class to impute missing values in a dataframe using XGBoost and Optuna for hyperparameter optimization.

    Attributes:
        df: The dataframe containing the data for imputation.
        OPTUNA: A boolean indicating whether to use Optuna for hyperparameter optimization.
    """

    def __init__(self, df, OPTUNA):
        """
        Initializes the Imputer instance with a dataframe and Optuna flag.

        Args:
            df: The dataframe containing the data for imputation.
            OPTUNA: Boolean flag to determine if Optuna is used.
        """
        self.df = df
        self.OPTUNA = OPTUNA
        return None

    def objective(self, trial):
        """
        Defines the Optuna objective function for optimizing the XGBoost classifier.

        Args:
            trial: The Optuna trial object for suggesting hyperparameters.

        Returns:
            The training score of the XGBoost model.
        """
        X_train, y_train, _ = self.get_x_and_y()

        param = {
            "objective": "multi:softmax",
            "num_class": len(self.label_mapping),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "random_state": 42,
        }

        model = xgboost.XGBClassifier(**param)
        model.fit(X_train, y_train)

        score = model.score(X_train, y_train)
        return score

    def get_x_and_y(self):
        """
        Prepares training and testing datasets and maps review summaries to numerical labels.

        Returns:
            A tuple containing:
                - Features for training (X_train).
                - Labels for training (y_train).
                - Dataframe for testing (test_data).
        """
        data = self.df.copy()

        train_data = data[data['review_summary'] != '']
        test_data = data[data['review_summary'] == '']
        self.label_mapping = {category: idx for idx, category in enumerate(train_data["review_summary"].unique())}

        train_data = train_data[['Reviews Total', 'review_score', 'launch_price', 'review_summary']]
        test_data = test_data[['Reviews Total', 'review_score', 'launch_price', 'review_summary']]

        train_data["review_summary"] = train_data["review_summary"].map(self.label_mapping)

        return (
            train_data[['Reviews Total', 'review_score', 'launch_price']],
            train_data['review_summary'].astype(int),
            test_data,
        )

    def optuna_trials(self):
        """
        Runs Optuna optimization to find the best hyperparameters for the XGBoost model.

        Returns:
            A dictionary of the best hyperparameters found by Optuna.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100)

        best_params = study.best_params
        best_params["objective"] = "multi:softmax"
        best_params["num_class"] = len(self.label_mapping)
        best_params["random_state"] = 42

        return best_params

    def train_model(self):
        """
        Trains an XGBoost model using either Optuna-optimized or default parameters.

        Returns: 
            A tuple containing:
                - The trained XGBoost model.
                - The test dataset for predictions.
        """
        if self.OPTUNA:
            best_params = self.optuna_trials()
        else:
            best_params = {
                'learning_rate': 0.2789690140794501,
                'max_depth': 10,
                'min_child_weight': 5,
                'subsample': 0.7184437293302535,
                'colsample_bytree': 0.6679592475294873,
                'n_estimators': 412,
                'lambda': 1.2641291472985302e-07,
                'alpha': 0.07583611487593768,
                'objective': 'multi:softmax',
                'num_class': 9,
                'random_state': 42
            }

        X_train, y_train, test_data = self.get_x_and_y()

        model = xgboost.XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        return model, test_data

    def predict_data(self):
        """
        Predicts missing values for the 'review_summary' column using the trained model.

        Returns:
            The updated dataframe with imputed 'review_summary' values.
        """
        # Training the model
        model, test_data = self.train_model()

        # Predicting the values
        predictions = model.predict(test_data[['Reviews Total', 'review_score', 'launch_price']])

        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        test_data["review_summary"] = [reverse_mapping[p] for p in predictions]

        self.df.loc[self.df["review_summary"] == "", "review_summary"] = test_data["review_summary"]
        return self.df
