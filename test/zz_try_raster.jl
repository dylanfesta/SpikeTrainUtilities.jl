push!(LOAD_PATH, abspath(@__DIR__,".."))
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Makie,CairoMakie
using Random
using Statistics
Random.seed!(0)


##

# helper for get_line_segments function
function _line_segments_onetrain(train::Vector{<:Real},offset::Real,heigth_scaling::Real)
  n_units = length(train)
  if n_units == 0
    # returns empty vector
    return Vector{Tuple{Float64,Float64}}()
  end
  npoints = n_units*2
  t_rep = repeat(train;inner=2)
  xs_nofix = fill(offset,npoints)
  fixes = repeat([-0.5*heigth_scaling,0.5*heigth_scaling];outer=n_units)
  xs_fix = xs_nofix .+ fixes
  return collect(zip(t_rep,xs_fix))
end

"""
  function get_line_segments(spiketrains::U.SpikeTrains,
    t_start::Real,t_end::Real;
    heigth_scaling::Real=0.7,
    neurons::Union{Vector{Int},Nothing}=nothing,
    offset::Real = 0.0,
    max_spikes::Real=1E6)

Returns point coordinates that can be used for line segments for a raster plot of the spiketrain

# Arguments
- `spiketrains::U.SpikeTrains` : the spiketrains
- `t_start::Real` : the starting time of the raster
- `t_end::Real` : the ending time of the raster
- `heigth_scaling::Real=0.7` : how tall each spike is, 1.0 is the full height
- `neurons::Union{Vector{Int},Nothing}=nothing` : the neurons to plot, if nothing all neurons are plotted
- `offset::Real=0.0` : the y-offset of the raster
- `max_spikes::Real=1E6` : the maximum number of spikes that can be plotted.

# Returns
- points_linesegment : a vector of tuples of two floats, (t1,y1a),(t1,y1b),(t2,y2a),(t2,y2b),... 
   where t1,t2 are the times of the spikes and y1a,y1b,y2a,y2b are the y-coordinates of the line segments,
   representing neurons on the y axis. This can be used directly as argument to `linesegments!` in Makie. 

"""
function get_line_segments(spiketrains::U.SpikeTrains,
    t_start::Real,t_end::Real;
    heigth_scaling::Real=0.7,
    neurons::Union{Vector{Int},Nothing}=nothing,
    offset::Real = 0.0,
    max_spikes::Real=1E6)

  ntot = spiketrains.n_units
  @assert t_start >= spiketrains.t_start "starting time $t_start is before the trains starting time $(spiketrains.t_start)"
  @assert t_end <= spiketrains.t_end "ending time $t_end is after the trains ending time $(spiketrains.t_end)"
  if !isnothing(neurons)
    @assert all(1 .<= neurons .<= ntot) "neurons must be between 1 and $ntot"
  else
    neurons = 1:ntot
  end
  nneus = length(neurons)

  trains_selected = map(neurons) do neu
    train = spiketrains.trains[neu]
    train_selected = filter(t->t_start<=t<=t_end,train)
    train_selected
  end
  tot_spikes = sum(length.(trains_selected))
  if tot_spikes > max_spikes
    # check rates, for safety
    numerical_rates = U.numerical_rates(spiketrains; t_start=t_start,t_end=t_end)
    error(" There are $nneus neurons with a mean rate of $(mean(numerical_rates)) Hz.
      The total number of spikes is $(tot_spikes) which is larger than the maximum $(max_spikes).
      Set a smaller time window, or fewer neruons.")
  end
  points_all = map(enumerate(trains_selected)) do (k,train)
    _line_segments_onetrain(train,offset+k,heigth_scaling)
  end
  return vcat(points_all...)
end


##

rates = [5.,10.0,5.,100.0,1000000.0]
T = 5.0
testspiketrains = U.make_random_spiketrains(rates,T)

points_linesegments = get_line_segments(testspiketrains,0.0,5.0;
  neurons=[2,3],
  heigth_scaling=0.95)

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

