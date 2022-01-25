from metaflow import step, current, FlowSpec, Parameter, card

class NBFlow(FlowSpec):

    exclude_nb_input = Parameter('exclude_nb_input', default=True, type=bool)

    @step
    def start(self):
        self.data_for_notebook = "I Will Print Myself From A Notebook"
        self.next(self.end)
    
    @card(type='notebook')
    @step
    def end(self):
        self.nb_options_dict = dict(input_path='nbflow.ipynb', exclude_input=self.exclude_nb_input)

if __name__ == '__main__':
    NBFlow()
