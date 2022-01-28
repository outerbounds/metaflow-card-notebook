from metaflow import step, current, FlowSpec, Parameter, card, resources
import os


class NBFlow(FlowSpec):
    """
    MLCode is taken from this tutorial: https://keras.io/examples/nlp/text_classification_from_scratch/
    """
    @classmethod
    def get_tune_data(cls, search_clf):
        """Extract results of hyperparameter tuning as a dataframe."""
        import pandas as pd
        tune_data = search_clf.cv_results_
        tune_viz = pd.DataFrame(tune_data['params'])
        tune_viz['neg_log_loss'] = tune_data['mean_test_score']
        return tune_viz


    @card(type='html')
    @step
    def start(self): 
        """Get the data and profile it."""
        import pandas as pd
        from pandas_profiling import ProfileReport
        self.raw_data = pd.read_csv('https://raw.githubusercontent.com/outerbounds/.data/main/hospital_readmission.csv')
        profile = ProfileReport(self.raw_data, title="Data Profile")
        self.html = profile.to_html()
        self.next(self.train_model)

    @resources(memory=40000, cpu=4)
    @step
    def train_model(self):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        import numpy as np
        import pandas as pd

        y = self.raw_data.readmitted
        X = pd.DataFrame(self.raw_data).drop(['readmitted'], axis=1)
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.extend([3,4,5])
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

        rf = RandomForestClassifier(max_features='auto')
        rf_search = RandomizedSearchCV(estimator=rf, 
                                    scoring = "neg_log_loss",
                                    param_distributions=random_grid, 
                                    n_iter=40, 
                                    cv=3, 
                                    random_state=42, 
                                    n_jobs=-1)

        # Save the data and the models
        self.train_X, self.val_X, self.train_y, self.val_y = train_X, val_X, train_y, val_y
        self.cv_clf = rf_search.fit(train_X, train_y)
        self.best_model = self.cv_clf.best_estimator_
        self.best_params_idx = self.cv_clf.best_index_
        self.tune_data = self.get_tune_data(self.cv_clf)
        self.next(self.evaluate)


    @card(type='notebook')
    @step
    def evaluate(self):
        self.nb_options_dict = dict(input_path='notebooks/Evaluate_Model.ipynb')
        self.next(self.end)


    @step
    def end(self): 
        pass

if __name__ == '__main__':
    NBFlow()
