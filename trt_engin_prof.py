import matplotlib.pyplot as plt
import os
import pandas as pd
from trex import *

# Configure a wider output (for the wide graphs)
set_wide_display()

# Choose an engine file to load.  This notebook assumes that you've saved the engine to the following paths.
engine_name = "weights/depth_anything_vits14-sim-ptq-f16"
print(engine_name)
assert engine_name is not None
plan = EnginePlan(f'{engine_name}.graph.json', f'{engine_name}.profile.json')
print(plan)
print(f"Summary for {plan.name}:\n")
plan.summary()
df = plan.df
display_df(plan.df)

layer_types = group_count(plan.df, 'type')

# Simple DF print
print(layer_types)

# dtale DF display
display_df(layer_types)

plotly_bar2(
    df=layer_types, 
    title='Layer Count By Type', 
    values_col='count', 
    names_col='type',
    orientation='v',
    color='type',
    colormap=layer_colormap,
    show_axis_ticks=(True, True))

top3 = plan.df.nlargest(3, 'latency.pct_time')
display_df(top3)

plotly_bar2(
    df=plan.df, 
    title="% Latency Budget Per Layer",
    values_col="latency.pct_time",
    names_col="Name",
    color='type',
    use_slider=False,
    colormap=layer_colormap)

plotly_hist(
    df=plan.df, 
    title="Layer Latency Distribution", 
    values_col="latency.pct_time",
    xaxis_title="Latency (ms)",
    color='type',
    colormap=layer_colormap)

fig = px.treemap(
    plan.df,
    path=['type', 'Name'],
    values='latency.pct_time',
    title='Treemap Of Layer Latencies (Size & Color Indicate Latency)',
    color='latency.pct_time')
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()

# fig = px.treemap(
#     plan.df,
#     path=['type', 'Name'],
#     values='latency.pct_time',
#     title='Treemap Of Layer Latencies (Size Indicates Latency. Color Indicates Activations Size)',
#     color='total_io_size_bytes')
# fig.update_traces(root_color="white")
# fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
# fig.show()

plotly_bar2(
    plan.df, 
    "Weights Sizes Per Layer", 
    "weights_size", "Name", 
    color='type', 
    colormap=layer_colormap)

plotly_bar2(
    plan.df, 
    "Activations Sizes Per Layer", 
    "total_io_size_bytes", 
    "Name", 
    color='type', 
    colormap=layer_colormap)

plotly_hist(
    plan.df, 
    "Layer Activations Sizes Distribution", 
    "total_io_size_bytes", 
    "Size (bytes)", 
    color='type', 
    colormap=layer_colormap)

plan.df["total_io_size_bytes"].describe()


charts = []
layer_precisions = group_count(plan.df, 'precision')
charts.append((layer_precisions, 'Layer Count By Precision', 'count', 'precision'))

layers_time_pct_by_precision = group_sum_attr(plan.df, grouping_attr='precision', reduced_attr='latency.pct_time')
display(layers_time_pct_by_precision)

charts.append((layers_time_pct_by_precision, '% Latency Budget By Precision', 'latency.pct_time', 'precision'))
plotly_pie2("Precision Statistics", charts, colormap=precision_colormap)


plotly_bar2(
    plan.df, 
    "% Latency Budget Per Layer<BR>(bar color indicates precision)", 
    "latency.pct_time", 
    "Name",
    color='precision',
    colormap=precision_colormap)

formatter = layer_type_formatter if True else precision_formatter
graph = to_dot(plan, formatter)
svg_name = render_dot(graph, engine_name, 'svg')
png_name = render_dot(graph, engine_name, 'png')
from IPython.display import Image
display(Image(filename=png_name))

convs1 = plan.df.query("type == 'Convolution'")
convs2 = df[df.type == 'Convolution']

convs = plan.get_layers_by_type('Convolution')
display_df(convs)

plotly_bar2(
    convs, 
    "Latency Per Layer (%)<BR>(bar color indicates precision)",
    "attr.arithmetic_intensity", "Name",
    color='precision', 
    colormap=precision_colormap)

plotly_bar2(
    convs,
    "Convolution Data Sizes<BR>(bar color indicates latency)",
    "total_io_size_bytes", 
    "Name", 
    color='latency.pct_time')

plotly_bar2(
    convs, 
    "Convolution Arithmetic Intensity<BR>(bar color indicates activations size)",
    "attr.arithmetic_intensity", 
    "Name",
    color='total_io_size_bytes')

plotly_bar2(
    convs, 
    "Convolution Arithmetic Intensity<BR>(bar color indicates latency)", 
    "attr.arithmetic_intensity", 
    "Name",
    color='latency.pct_time')

# Memory accesses per ms (assuming one time read/write penalty)
plotly_bar2(
    convs, 
    "Convolution Memory Efficiency<BR>(bar color indicates latency)", 
    "attr.memory_efficiency", 
    "Name", 
    color='latency.pct_time')

# Compute operations per ms (assuming one time read/write penalty)
plotly_bar2(
    convs, 
    "Convolution Compute Efficiency<BR>(bar color indicates latency)",
    "attr.compute_efficiency",
    "Name",
    color='latency.pct_time')


convs = plan.get_layers_by_type('Convolution')

charts = []
convs_count_by_type = group_count(convs, 'subtype')
charts.append((convs_count_by_type, 'Count', 'count', 'subtype'))

convs_time_pct_by_type = group_sum_attr(convs, grouping_attr='subtype', reduced_attr='latency.pct_time')
charts.append((convs_time_pct_by_type, '% Latency Budget', 'latency.pct_time', 'subtype'))
plotly_pie2("Convolutions Statistics (Subtype)", charts)


charts = []
convs_count_by_group_size = group_count(convs, 'attr.groups')
charts.append((convs_count_by_group_size, 'Count', 'count', 'attr.groups'))

convs_time_pct_by_grp_size = group_sum_attr(convs, grouping_attr='attr.groups', reduced_attr='latency.pct_time')
charts.append((convs_time_pct_by_grp_size, '% Latency Budget', 'latency.pct_time', 'attr.groups'))
plotly_pie2("Convolutions Statistics (Number of Groups)", charts)



charts = []
convs_count_by_kernel_shape = group_count(convs, 'attr.kernel')
charts.append((convs_count_by_kernel_shape, 'Count', 'count', 'attr.kernel'))

convs_time_pct_by_kernel_shape = group_sum_attr(convs, grouping_attr='attr.kernel', reduced_attr='latency.pct_time')
charts.append((convs_time_pct_by_kernel_shape, '% Latency Budget', 'latency.pct_time', 'attr.kernel'))
plotly_pie2("Convolutions Statistics (Kernel Size)", charts)


charts = []
convs_count_by_precision = group_count(convs, 'precision')
charts.append((convs_count_by_precision, 'Count', 'count', 'precision'))

convs_time_pct_by_precision = group_sum_attr(convs, grouping_attr='precision', reduced_attr='latency.pct_time')
charts.append((convs_time_pct_by_precision, '% Latency Budget', 'latency.pct_time', 'precision'))

plotly_pie2("Convolutions Statistics (Precision)", charts, colormap=precision_colormap)