push!(LOAD_PATH, abspath(@__DIR__,".."))
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Makie,CairoMakie
using Random
using Statistics
Random.seed!(0)


##

rates = [5.,10.0,5.,100.0,1000000.0]
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
##

f = Figure()
Axis(f[1, 1])

xs = 1:0.2:10
ys = sin.(xs)

linesegments!(xs, ys)
display(f)
##
f = let _f=Figure()
  ax=Axis(_f[1, 1])
  plot_points_test = [ (Point(1,0.0),Point(1,1.0) ), (Point(2.1,0.0),Point(2.1,1.0) ) ]
  plot_points_test2 = [ (1,0.0), (1.1,1), (2.1,0.0), (2.1,1.0)]
  testthis = _line_segments_onetrain([1.,2.,4.],0.0,0.7)
  testthat = _line_segments_onetrain([0.,3.,6.],1.0,0.7)
  testt3 = _line_segments_onetrain(Vector{Float64}(),3.0,0.7)
  _test = vcat(testthis,testthat,testt3)
  linesegments!(ax,_test)
  _f
end
display(f)

