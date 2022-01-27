from metaflow import step, current, FlowSpec, Parameter, conda_base, conda, card, resources
import os


@conda_base(python='3.8.10', libraries={'pandas':'1.3.5', 'scikit-learn':'1.0.2', 'matplotlib':'3.1.1', 'numpy':'1.22.0'})
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

    @conda(libraries={'pandas-profiling': '3.1.0'}) # 'metaflow-card-html':'1.0.0'
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

    # 'metaflow-card-notebook':'1.0.1', 'papermill':'2.3.3'
    @resources(memory=10000, cpu=3)
    @conda(libraries={'pdpbox':'0.2.1', 'altair':'4.2.0', 'plotly':'5.5.0'})
    @card(type='notebook')
    @step
    def evaluate(self):
        os.system('pip install metaflow-card-notebook')
        self.nb_options_dict = dict(input_path='notebooks/Evaluate_Model.ipynb')
        self.next(self.end)


    @step
    def end(self): 
        pass

if __name__ == '__main__':
    NBFlow()
