from metaflow import step, current, FlowSpec, Parameter, card
from fastcore.all import run

class NBFlow(FlowSpec):
    """
    Training a model in Metaflow with notebooks.  
    MLCode is taken from this tutorial: https://keras.io/examples/nlp/text_classification_from_scratch/
    """
    batch_size = Parameter('batch_size', type=int, default=64)
    seed = Parameter('seed', type=int, default=1337)
    max_features = Parameter('max_features', type=int, default=10000)
    embedding_dim = Parameter('embedding_dim', type=int, default=64)
    sequence_length = Parameter('sequence_length', type=int, default=500)
    num_epochs = Parameter('num_epochs', type=int, default=3)

    @step
    def start(self): 
        """Get the data."""
        from classify import get_data
        self.raw_data = get_data(url='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
        self.next(self.train)

    
    @step
    def train(self):
        """Train the model(s)."""
        from classify import build_model, train_model
        model=build_model(train_x=self.raw_data['train'][0], 
                               embedding_dim=self.embedding_dim,
                               max_features=self.max_features,
                               sequence_length=self.sequence_length)
        
        self.model_results = train_model(model=model, data=self.raw_data, num_epochs=self.num_epochs, batch_size=self.batch_size)
        self.next(self.nb_auto)
    

    @card(type='notebook')
    @step
    def nb_auto(self):
        """Run & Render Jupyter Notebook With The Notebook Card."""
        self.nb_options_dict = dict(input_path='notebooks/Evaluate.ipynb')
        self.next(self.nb_manual)


    @card(type='html')
    @step
    def nb_manual(self):
        """
        Run & Render Jupyter Notebook Manually With The HTML Card.
        
        Using the html card provides you greater control over notebook execution and rendering.
        """
        import papermill as pm
        output_nb_path = 'notebooks/rendered_Evaluate.ipynb'
        output_html_path = output_nb_path.replace('.ipynb', '.html')

        pm.execute_notebook('notebooks/Evaluate.ipynb',
                            output_nb_path,
                            parameters=dict(pathspec = current.pathspec)
                            )
        run(f'jupyter nbconvert --to html --no-input --no-prompt {output_nb_path}')
        with open(output_html_path, 'r') as f:
            self.html = f.read()
        self.next(self.end)
    

    @step
    def end(self): 
        pass

if __name__ == '__main__':
    NBFlow()
