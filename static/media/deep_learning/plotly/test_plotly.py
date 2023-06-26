import plotly
from plotly.graph_objs import *
import plotly.graph_objects as go
import numpy as np

# # Create figure
# fig = go.Figure()

# # Add traces, one for each slider step
# for step in np.arange(0, 5, 0.1):
#     fig.add_trace(
#         go.Scatter(
#             visible=False,
#             line=dict(color="#00CED1", width=6),
#             name="𝜈 = " + str(step),
#             x=np.arange(0, 10, 0.01),
#             y=np.sin(step * np.arange(0, 10, 0.01))
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             visible=False,
#             line=dict(color="#00CED1", width=6),
#             name="𝜈 = " + str(step),
#             x=np.arange(0, 10, 0.01),
#             y=np.cos(step * np.arange(0, 10, 0.01))
#         )
#     )

# # Make 10th trace visible
# fig.data[10].visible = True

# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="update",
#         args=[{"visible": [False] * len(fig.data)},
#               {"title": "Slider switched to step: " + str(i)}],  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Frequency: "},
#     pad={"t": 100},
#     steps=steps
# )]

f    = lambda x: ((x**4)+(x**2)+(10*x))/50
g    = lambda x: ((4*x**3)+(2*x)+10)/50
fnew = lambda x: f(x) - (x*g(x)**2)


num_steps = 10
trace_list1 = []
trace_list2 = []
trace_list3 = []
start  = -3
finish = 3
xfull = np.arange(-5,5,0.01)
yfull = f(xfull)
for i in range(num_steps):
    if(i==0):
        vis = True
    else:
        vis = False
    x = np.arange(-3,3,(finish-start)/num_steps)
    y = f(x)
    gr = g(x)
    trace_list1.append(go.Scatter(x=xfull,y=yfull, visible=vis, line={'color': 'red'}))
    trace_list2.append(go.Scatter(x=xfull,y=g(x[i])*(xfull-x[i])+f(x[i]), visible=vis, line={'color': 'blue'}))
    trace_list3.append(go.Scatter(x=[x[i]],y=[y[i]], visible=vis, line={'color': 'green'}))

fig = go.Figure(data=trace_list1+trace_list2+trace_list3)

steps = []
for i in range(num_steps):
    # Hide all traces
    step = dict(
        method = 'restyle',  
        args = ['visible', [False] * len(fig.data)],
    )
    # Enable the two traces we want to see
    step['args'][1][i] = True           # Add figure from list 1
    step['args'][1][i+num_steps] = True # Add figure from list 2
    step['args'][1][i+2*num_steps] = True
    
    # Add step to step list
    steps.append(step)

sliders = [dict(
    steps = steps,
)]

fig.layout.sliders = sliders
fig['layout']['sliders'][0]['pad']=dict(r= 0, t= 50,)
fig['layout']['margin']=dict(t=20)

fig.update_layout(
    sliders=sliders,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False,
    xaxis_range=[-3,3],
    yaxis_range=[-1,3],
    xaxis=dict(showgrid=False,mirror=True,ticks='outside',showline=True,zeroline=False),
    yaxis=dict(showgrid=False,mirror=True,ticks='outside',showline=True,zeroline=False,fixedrange=False),
    xaxis_title=r"<i>x</i>", yaxis_title=r"<i>f(x)</i>",
    # font_family="Serif"

)

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

plotly.io.write_json(fig=fig,file='./test_file.json')
# fig.show()