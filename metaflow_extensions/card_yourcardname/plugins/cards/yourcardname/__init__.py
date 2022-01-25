from metaflow.cards import MetaflowCard

class YourCard(MetaflowCard):

    type = 'your_card'
    
    def __init__(self, options={"attribute":"html"}, **kwargs):
        self._attr_nm = options.get("attribute", "html")
 
    def render(self, task):
        if self._attr_nm in task:
            return str(task[self._attr_nm].data)

CARDS = [YourCard]