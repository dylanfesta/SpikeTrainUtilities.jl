

"""
  draw_spike_raster(trains::Vector{Vector{Float64}},
      dt::Real,Tend::Real;
      Tstart::Real=0.0,
      spike_size::Integer = 5,
      spike_separator::Integer = 1,
      background_color::Color=RGB(1.,1.,1.),
      spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
      max_size::Real=1E4) where C<:Color

Draws a matrix that contains the raster plot of the spike train.

# Arguments
  + `Trains` :  Vector of spike trains. The order of the vector corresponds to 
    the order of the plot. First element is at the top, second is second row, etc.
  + `dt` : time interval representing one horizontal pixel  
  + `Tend` : final time to be considered

# Optional arguments
  + `Tstart::Real` : starting time
  + `spike_size::Integer` : heigh of spike (in pixels)
  + `spike_separator::Integer` : space between spikes, and vertical padding
  + `background_color::Color` : self-explanatory
  + `spike_colors::Union{Color,Vector{Color}}` : if a single color, color of all spikes, if vector of colors, 
     color for each neuron (length should be same as number of neurons)
  + `max_size::Integer` : throws an error if image is larger than this number (in pixels)

# Returns
  + `raster_matrix::Matrix{Color}` you can save it as a png file
"""
function draw_spike_raster(trains::Vector{Vector{Float64}},
  dt::Real,time_duration::Real;
    t_start::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where C<:Color
  nneus = length(trains)
  t_end = t_start + time_duration
  if typeof(spike_colors) <: Color
    spike_colors = repeat([spike_colors,];outer=nneus)
  else
    @assert length(spike_colors) == nneus "error in setting colors"
  end
  binned_binary  = map(trains) do train
    .! iszero.(bin_spikes(train,dt,t_end;t_start=t_start)[2])
  end
  ntimes = length(binned_binary[1])
  ret = fill(background_color,
    (nneus*spike_size + # spike sizes
      spike_separator*nneus + # spike separators (incl. top/bottom padding) 
      spike_separator),ntimes)
  @assert all(size(ret) .< max_size ) "The image is too big! Please change time limits"  
  for (neu,binv,col) in zip(1:nneus,binned_binary,spike_colors)
    spk_idx = findall(binv)
    _idx_pre = (neu-1)*(spike_size+spike_separator)+spike_separator
    y_color = _idx_pre+1:_idx_pre+spike_size
    ret[y_color,spk_idx] .= col
  end
  return ret
end


function draw_spike_raster(S::SpikeTrains{R,N},dt::Real,time_duration::Real;
    t_start::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where {R,N,C<:Color}
  @assert t_start >= S.t_start "t_start is too small"   
  @assert t_start + time_duration <= S.t_end "t_start + duration is too large"
  trains = S.trains
  return draw_spike_raster(trains,dt,time_duration;
    t_start=t_start,
    spike_size=spike_size,
    spike_separator=spike_separator,
    background_color=background_color,
    spike_colors=spike_colors,
    max_size=max_size)
end


function plot_spike_raster(args...)
  @error "Load Makie and CairoMakie first!"
  return nothing
end




#=
Example of raster plot with Makie

f = let f= Figure()
  dt_rast=0.0005
  Δt_rast = 0.50
  Tstart_rast = 100.
  Tend_rast = Tstart_rast + Δt_rast
  rast_spk = 10
  rast_sep = 3
  theraster=H.draw_spike_raster(trains,dt_rast,Tend_rast;Tstart=Tstart_rast,
  spike_size=rast_spk,spike_separator=rast_sep)
  rast = permutedims(theraster)
  neu_yvals  = let n_neu = n_default,
    N = 2*n_neu*(rast_sep+rast_spk)+2*rast_sep+rast_spk
    ret = collect(range(0,n_neu+1;length=N))
    ret[rast_spk+1:end-rast_spk]
  end
  pxtimes,pxneus = size(rast)
  times = range(0,Δt_rast,length=pxtimes)
  ax1 = Axis(f[1, 1], aspect=pxtimes/pxneus,
    rightspinevisible=false,topspinevisible=false,
    xlabel="time (s)",ylabel="neuron #")
  image!(ax1,times,neu_yvals,rast)
  f
end
=#


# helper for get_line_segments function
function _line_segments_onetrain(train::Vector{<:Real},
  neuron_offset::Real,time_offset::Real,height_scaling::Real)
  n_units = length(train)
  if n_units == 0
    # returns empty vector
    return Vector{Tuple{Float64,Float64}}()
  end
  npoints = n_units*2
  t_rep = repeat(train;inner=2)
  t_rep .-= time_offset
  xs_nofix = fill(neuron_offset,npoints)
  fixes = repeat([-0.5*height_scaling,0.5*height_scaling];outer=n_units)
  xs_fix = xs_nofix .+ fixes
  return collect(zip(t_rep,xs_fix))
end

"""
  function get_line_segments(spiketrains::SpikeTrains,
    t_start::Real,t_end::Real;
    height_scaling::Real=0.7,
    neurons::Union{Vector{Int},Nothing}=nothing,
    offset::Real = 0.0,
    max_spikes::Real=1E6)

Returns point coordinates that can be used for line segments for a raster plot of the spiketrain

# Arguments
- `spiketrains::SpikeTrains` : the spiketrains
- `t_start::Real` : the starting time of the raster
- `t_end::Real` : the ending time of the raster
- `height_scaling::Real=0.7` : how tall each spike is, 1.0 is the full height
- `neurons::Union{Vector{Int},Nothing}=nothing` : the neurons to plot, if nothing all neurons are plotted
- `neuron_offset::Real=0.0` : the y-offset of the raster
- `time_offset::Real` : the time offset of the raster, default is `t_start`
- `max_spikes::Real=1E6` : the maximum number of spikes that can be plotted.

# Returns
- points_linesegment : a vector of tuples of two floats, (t1,y1a),(t1,y1b),(t2,y2a),(t2,y2b),... 
   where t1,t2 are the times of the spikes and y1a,y1b,y2a,y2b are the y-coordinates of the line segments,
   representing neurons on the y axis. This can be used directly as argument to `linesegments!` in Makie. 

"""
function get_line_segments(spiketrains::SpikeTrains,
    t_start::Real,t_end::Real;
    height_scaling::Real=0.7,
    neurons::Union{Vector{Int},Nothing}=nothing,
    neuron_offset::Real = 0.0,
    time_offset::Union{Real,Nothing}=nothing,
    max_spikes::Real=1E6)

  ntot = spiketrains.n_units
  @assert t_start >= spiketrains.t_start "starting time $t_start is before the trains starting time $(spiketrains.t_start)"
  @assert t_end <= spiketrains.t_end "ending time $t_end is after the trains ending time $(spiketrains.t_end)"
  if !isnothing(neurons)
    @assert all(1 .<= neurons .<= ntot) "neurons must be between 1 and $ntot"
  else
    neurons = 1:ntot
  end
  if !isnothing(time_offset)
    @assert time_offset >= t_end "time_offset is probably wrong"
  else
    time_offset = t_start
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
    numerical_rates = numerical_rates(spiketrains; t_start=t_start,t_end=t_end)
    error(" There are $nneus neurons with a mean rate of $(mean(numerical_rates)) Hz.
      The total number of spikes is $(tot_spikes) which is larger than the maximum $(max_spikes).
      Set a smaller time window, or fewer neruons.")
  end
  points_all = map(enumerate(trains_selected)) do (k,train)
    _line_segments_onetrain(train,neuron_offset+k,time_offset,height_scaling)
  end
  return vcat(points_all...)
end

