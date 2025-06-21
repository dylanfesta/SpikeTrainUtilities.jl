#=

Raster plot, using PlotlyJS instead of Makie

=#

push!(LOAD_PATH, abspath(@__DIR__,".."))
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using PlotlyJS
using Random
using Distributions
Random.seed!(0);

##

n=50
the_rates = 20 .* rand(n)
T = 60.0

spiketrains = U.make_random_spiketrains(the_rates,T)

spiketrains_raster = U.get_line_segments(spiketrains,0.0,T;
    height_scaling=0.7,
    max_spikes=10E6)

x_plot,y_plot = U.line_segments_to_xynans(spiketrains_raster)    

# raster_plotly = reshape(spiketrains_raster,2,:)
# line_add = [ (xy[1],NaN) for xy in raster_plotly[2,:]]

# raster_plotly = cat(raster_plotly,permutedims(line_add), dims=1)

# x_plot_all = get.(raster_plotly, 1, NaN)[:]
# y_plot_all = get.(raster_plotly, 2, NaN)[:]

plotly_trace = [ scatter(
    x = x_plot,
    y = y_plot,
    mode = "lines",
    line = attr(width=0.5, color="black"),
    name = "Spike raster"
), 
]
plotly_layout = Layout(
    title = "Raster Plot",
    xaxis = attr(title = "time (s)"),
    yaxis = attr(title = "neuron", dtick=1, range=[0.1,n+0.9]),
    showlegend = false,
    height = 600,
    width = 1000,
    plot_bgcolor = "white",
)

plotly_figure = Plot(plotly_trace, plotly_layout)
display(plotly_figure)

##