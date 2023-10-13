module SpikeTrainPlotting
using SpikeTrainUtilities
using Makie,CairoMakie
using Colors

function SpikeTrainUtilities.plot_spike_raster(trains::Vector{Vector{Float64}},
    dt::Real,time_duration::Real;
    t_start::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where C<:Color

  theraster = SpikeTrainUtilities.draw_spike_raster(trains,dt,time_duration;
    t_start=t_start,
    spike_size=spike_size,
    spike_separator=spike_separator,
    background_color=background_color,
    spike_colors=spike_colors,
    max_size=max_size)

  f = Figure()
  rast = permutedims(theraster)
  pxtimes,pxneus = size(rast)
  neu_yvals  = let n_neu = length(trains),
    N = 2*n_neu*(spike_separator+spike_size)+2*spike_separator+spike_size
    ret = collect(range(0,n_neu+1;length=N))
    ret[spike_size+1:end-spike_size]
  end
  pxtimes,pxneus = size(rast)
  times = range(0,time_duration,length=pxtimes)
  ax1 = Axis(f[1, 1], aspect=pxtimes/pxneus,
    rightspinevisible=false,topspinevisible=false,
    xlabel="time (s)",ylabel="neuron #")
  image!(ax1,times,neu_yvals,rast)
  return f
end


function SpikeTrainUtilities.plot_spike_raster(S::SpikeTrains{R,N},dt::Real,time_duration::Real;
    t_start::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where {R,N,C<:Color}
  @assert t_start >= S.t_start "t_start is too small"   
  @assert t_start + time_duration <= S.t_end "t_start + duration is too large"
  trains = S.trains
  return SpikeTrainUtilities.plot_spike_raster(trains,dt,time_duration;
    t_start=t_start,
    spike_size=spike_size,
    spike_separator=spike_separator,
    background_color=background_color,
    spike_colors=spike_colors,
    max_size=max_size)
end

end # module