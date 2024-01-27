from trex import *
engine_name = "weights/depth_anything_vits14-sim-ptq-f16"
print(engine_name)
assert engine_name is not None
plan = EnginePlan(f'{engine_name}.graph.json', f'{engine_name}.profile.json')
formatter = layer_type_formatter if True else precision_formatter
graph = to_dot(plan, formatter)
svg_name = render_dot(graph, engine_name, 'svg')
png_name = render_dot(graph, engine_name, 'png')
from IPython.display import Image
display(Image(filename=png_name))
