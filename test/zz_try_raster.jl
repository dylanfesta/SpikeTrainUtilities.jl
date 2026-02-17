push!(LOAD_PATH, abspath(@__DIR__,".."))
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Makie,CairoMakie,Colors,NamedColors
using Random
using Statistics
Random.seed!(0)


##

rates = [5.,10.0,5.,100.0,200.0]
T = 10.0
testspiketrains = U.make_random_spiketrains(rates,T)

points_linesegments = U.get_line_segments(testspiketrains,1.0,6.0;
  neurons=[2,3],
  height_scaling=0.95)

##
f = let _f=Figure()
  ax=Axis(_f[1, 1])
  linesegments!(ax,points_linesegments)
  _f
end
display(f)


## Use old function
U.plot_spike_raster(testspiketrains,1E-3,2.0;
  t_start=1.0,
  spike_size=50,
  spike_separator=8,
  background_color=RGB(1.,1.,1.),
  spike_colors=RGB(0.,0.0,0.0),
  max_size=10E4)

##