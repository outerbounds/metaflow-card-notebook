from traitlets.config import Config
from metaflow.cards import MetaflowCard
from metaflow import current
import papermill as pm
from pathlib import Path
from nbconvert import HTMLExporter
import nbformat

class NotebookCard(MetaflowCard):

    type = 'notebook'

    def __init__(self, options={"input_path":None}, **kwargs):
        self.options = options
        
        if 'input_path' not in self.options or not self.options['input_path']:
            raise ValueError("Must specify 'input_path' in papermill_options in @card(type=notebook, papermill_options={'input_path':'directory/your_notebook.ipynb'}")
        else:
            self.input_path = Path(self.options['input_path'])
            if not self.input_path.name.endswith('.ipynb'):
                raise ValueError(f"input_path must be a notebook file, not {self.input_path}")
            if not self.input_path.exists():
                raise ValueError(f"Input notebook does not exist: {self.input_path}\n The current directory is {Path.cwd()}")

        c = Config()
        c.no_prompt = True
        c.no_input = True
        self.html_exporter = HTMLExporter(config=c, template_name = 'classic')
        self.run_id = current.run_id
        self.flow_name = current.flow_name
        
        # Calcualate output path and filename if none is given for the rendered notebook
        if 'output_path' not in self.options or not self.options['output_nb']:
            new_fn = f"_rendered_{self.run_id}_{self.input_path.name}"
            self.output_path = self.input_path.with_name(new_fn)
        else:
            self.output_path = self.options['output_nb']

        # Clean up papermill_options for any conflicting options
        for k in ['input_path', 'output_path']:
            self.options.pop(k, None)
        self.options['parameters'] = (self.options
                                                .get('parameters', {})
                                                .update(dict(run_id=self.run_id, flow_name=self.flow_name))
                                                 )
        
    def render(self, task):

        # Execute the notebook
        print(self.options)
        pm.execute_notebook(input_path=self.input_path,
                            output_path=self.output_path,
                            **self.options)
        
        # Render the notebook to HTML
        with open(self.output_path, 'r') as f:
            notebook = nbformat.reads(f, as_version=4)
            (body, resources) = self.html_exporter.from_notebook_node(notebook)
            return body

CARDS = [NotebookCard]
