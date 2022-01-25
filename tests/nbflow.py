from metaflow import step, current, FlowSpec, card

class NBFlow(FlowSpec):

    @step
    def start(self):
        self.varialbe_for_notebook = "I Will Print Myself From A Notebook"
        self.next(self.end)
    
    @card(type='notebook', options={'input_nb':'nbflow.ipynb'})
    @step
    def end(self):
        pass

if __name__ == '__main__':
    NBFlow()
