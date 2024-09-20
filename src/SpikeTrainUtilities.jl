module SpikeTrainUtilities
using LinearAlgebra,Statistics,StatsBase
using Colors
using RollingFunctions

"""
  SpikeTrains(n_units::N,trains::Vector{Vector{R}},t_start::R,t_end::R) where {R,N}

Constructs a SpikeTrains object.  
"""
mutable struct SpikeTrains{R,N}
  n_units::N
  trains::Vector{Vector{R}}
  t_start::R
  t_end::R
end 

function SpikeTrains(trains::Vector{Vector{R}}; 
      t_start::Union{R,Nothing}=nothing,t_end::Union{R,Nothing}=nothing) where {R}
  _last(v) = ifelse(isempty(v),-Inf,last(v))
  t_start = something(t_start,zero(R))
  t_end = something(t_end, maximum(_last,trains)+ 10*eps(R))
  n = length(trains)
  return SpikeTrains(n,trains,t_start,t_end)
end


# constructor from spiketimes and spikeneurons
function SpikeTrains(spiketimes::Vector{R},spikeneurons::Vector{I},
    nneus::Integer;t_start::Union{R,Nothing}=nothing,t_end::Union{R,Nothing}=nothing) where {R,I<:Integer}
  t_start = something(t_start,spiketimes[1] - 10*eps(R))
  t_end = something(t_end,spiketimes[end] + 10*eps(R))
  trains = Vector{Vector{R}}(undef,nneus)
  for neu in 1:nneus
    idxspike = spikeneurons .== neu
    if any(idxspike)
      trains[neu] = spiketimes[idxspike]
    else
      trains[neu] = Vector{R}()
    end 
  end
  return SpikeTrains(nneus,trains,t_start,t_end)
end

function delta_t(S::SpikeTrains{R,N}) where {R,N}
  return S.t_end - S.t_start
end


function Base.cat(S::SpikeTrains{R,N}...;dims::Int=1) where {R,N}
  @assert dims == 1 "dim can only be one"
  # check same number of units
  n_units = S[1].n_units
  for s in S
    if s.n_units != n_units
      error("All SpikeTrains must have same number of units")
    end
  end
  # check that times are consistent
  t_check = S[1].t_end
  for s in S[2:end]
    if !isapprox(t_check,s.t_start;atol=1e-4)
      error("SpikeTrains must be contiguous")
    end
    t_check = s.t_end
  end
  # okay, now cat
  t_start = S[1].t_start
  t_end = S[end].t_end
  trains = Vector{Vector{R}}(undef,n_units)
  for i in 1:n_units
    trains[i] = cat([s.trains[i] for s in S]...,dims=1)
  end
  return  SpikeTrains(n_units,trains,t_start,t_end)
end

@inline function Base.minimum(S::SpikeTrains{R,N}) where {R,N}
  _first(v) = ifelse(isempty(v),Inf,first(v))
  minimum(_first, S.trains)
end
@inline function Base.maximum(S::SpikeTrains{R,N}) where {R,N}
  _last(v) = ifelse(isempty(v),-Inf,last(v))
  maximum(last, S.trains)
end


# This is to merge them together, horizontally, so to speak
function Base.merge(S::SpikeTrains{R,N}...) where {R,N}
  ntrains = length(S)
  if ntrains == 1
    return S[1]
  end
  # check at least t_start is the same
  t_start = S[1].t_start
  @assert all(s->isapprox(s.t_start,t_start;atol=1e-4),S) "Starting times should be the same"
  new_tend = maximum(s->s.t_end,S)
  new_n_units = sum(s->s.n_units,S)
  new_trains = cat([s.trains for s in S]...,dims=1)
  @assert length(new_trains) == new_n_units "Something went wrong"
  return SpikeTrains(new_n_units,new_trains,t_start,new_tend)
end

## analysis here

# this is not optimized, one could use searchsortedfirst

@inline function count_spikes_in_interval(train::Vector{R},t_start::R,t_end::R) where R
  @assert t_start < t_end "t_start must be smaller than t_end"
  idx_first = searchsortedfirst(train,t_start)
  idx_last = searchsortedlast(train,t_end)
  return idx_last - idx_first + 1
end
@inline function numerical_rate(train::Vector{R},t_start::R,t_end::R) where R
  @assert t_start < t_end "t_start must be smaller than t_end"
  return count_spikes_in_interval(train,t_start,t_end)/(t_end-t_start)
end

"""
  numerical_rates(S::SpikeTrains{R,N};t_start::R=0.0,t_end::R=Inf) where {R,N}

Compute the numerical rates of the spike trains in `S` over the interval
 `[t_start,t_end]` if specified. If not, uses the interval `[S.t_start,S.t_end]`.
"""
function numerical_rates(S::SpikeTrains{R,N};
    t_start::Union{R,Nothing}=nothing,t_end::Union{R,Nothing}=nothing) where {R,N}
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  return [numerical_rate(train,t_start,t_end) for train in S.trains]
end

function count_spikes_in_interval(S::SpikeTrains{R,N},t_start::R,t_end::R) where {R,N}
  return [count_spikes_in_interval(train,t_start,t_end) for train in S.trains]
end



"""
    bin_spikes(Y::Vector{R},dt::R,Tend::R;Tstart::R=0.0) where R

# Arguments
  + `Y::Vector{<:Real}` : vector of spike times
  + `dt::Real` : time bin size
  + `Tend::Real` : end time for the raster
# Optional argument
  + `Tstart::Real=0.0` : start time for the raster

# Returns
  + times_bin::Vector{R} : `times_bin[k]` is the midpoint of the timebin `k` (i.e. `Tstart + (k-1/2)*dt`)   
  + `binned_spikes::Vector{<:Integer}` : `binned_spikes[k]` is the number of spikes that occur 
      in the timebin `k`  (i.e. between `Tstart + (k-1)*dt` and `Tstart + k*dt`)
"""
function bin_spikes(Y::Vector{R},dt::R,t_end::R;t_start::R=0.0) where R
  times = range(t_start,t_end;step=dt)  
  ret = fill(0,length(times)-1)
  for y in Y
    if t_start < y <= last(times)
      k = searchsortedfirst(times,y)-1
      ret[k] += 1
    end
  end
  return midpoints(times),ret
end

function bin_spikes(S::SpikeTrains{R,N},dt::R) where {R,N}
  nbins = Int64(div(delta_t(S),dt))
  times = range(S.t_start,S.t_end;length=nbins+1)
  binned = fill(0,(nbins,S.n_units))
  for (neu,train) in enumerate(S.trains)
    for y in train
      if S.t_start < y <= last(times)
        k = searchsortedfirst(times,y)-1
        binned[k,neu] += 1
      end
    end
  end
  return midpoints(times),binned
end

function instantaneous_rates(S::SpikeTrains{R,N},dt::R) where {R,N}
  times,binned = bin_spikes(S,dt)
  return times,binned ./ dt
end


function covariance_density_ij(X::Vector{R},Y::Vector{R},dτ::Real,τmax::R;
      t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where R
  times_cov = range(0.0,τmax-dτ;step=dτ)
  ndt = length(times_cov)
  t_end = something(t_end, max(X[end],Y[end])- dτ)
  t_start = something(t_start,max(0.0,min(X[1],Y[1]) - dτ))
  @assert t_start < t_end "t_start must be smaller than t_end"
  times_cov_ret = vcat(-reverse(times_cov[2:end]),times_cov)
  ret = Vector{Float64}(undef,2*ndt-1)
  binnedx = bin_spikes(X,dτ,t_end;t_start=t_start)[2]
  binnedy = bin_spikes(Y,dτ,t_end;t_start=t_start)[2]
  rx = numerical_rate(X,t_start,t_end) 
  ry = numerical_rate(Y,t_start,t_end) 
  ndt_tot = length(binnedx)
  binned_sh = similar(binnedx)
  # 0 and forward
  @simd for k in 0:ndt-1
    circshift!(binned_sh,binnedy,-k)
    ret[ndt-1+k+1] = dot(binnedx,binned_sh)
  end
  # backward
  @simd for k in 1:ndt-1
    circshift!(binned_sh,binnedy,k)
    ret[ndt-k] = dot(binnedx,binned_sh)
  end
  @. ret = (ret / (ndt_tot*dτ^2)) - rx*ry
  return times_cov_ret, ret
end


function covariance_density_ij(S::SpikeTrains{R,N},ij::Tuple{I,I},dτ::Real,τmax::R;
     t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where {R,N,I<:Integer}
  train_i = S.trains[ij[1]]
  train_j = S.trains[ij[2]]
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  return covariance_density_ij(train_i,train_j,dτ,τmax;t_end=t_end,t_start=t_start)
end


function variance_spike_count(X::Vector{R},Δt::R;
      t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing,
      warn_empty::Bool=true) where {R}
  # if empty, return 0.0
  if (isempty(X) && warn_empty)
    @warn "One of the spike trains is empty! Correlation set to 0.0"
    return 0.0
  end
  t_end = something(t_end,X[end]- Δt)
  t_start = something(t_start,max(0.0,X[1]- Δt))
  @assert t_start < t_end "t_start must be smaller than t_end"
  binnedx = bin_spikes(X,Δt,t_end;t_start=t_start)[2]
  return var(binnedx)
end

function variance_spike_count(S::SpikeTrains{R,N},i::I,Δt::R;
      t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where {R,N,I<:Integer}
  train_i = S.trains[i]
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  return variance_spike_count(train_i,Δt;t_end=t_end,t_start=t_start)
end

function fano_factor_spike_count(X::Vector{R},Δt::R;
         t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing,
         warn_empty::Bool=true) where {R}
  # if empty, return 0.0
  if isempty(X) && warn_empty
    @warn "The spike train is empty! Fano factor set to 0.0"
    return 0.0
  end
  t_end = something(t_end,X[end]- Δt)
  t_start = something(t_start,max(0.0,X[1]- Δt))
  @assert t_start < t_end "t_start must be smaller than t_end"
  binnedx = bin_spikes(X,Δt,t_end;t_start=t_start)[2]
  _var = var(binnedx)
  _mean = mean(binnedx)
  return _var/_mean
end

function fano_factor_spike_count(S::SpikeTrains{R,N},i::I,Δt::R;
      t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where {R,N,I<:Integer}
  train_i = S.trains[i]
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  return fano_factor_spike_count(train_i,Δt;t_end=t_end,t_start=t_start)
end

function coefficient_of_variation(X::Vector{R};
    t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where R
  t_start = something(t_start,X[1])
  t_end = something(t_end,X[end])
  idx_start = searchsortedfirst(X,t_start)
  idx_end = searchsortedlast(X,t_end)
  Xless = view(X,idx_start:idx_end)
  Xdiff = diff(Xless)
  return std(Xdiff)/mean(Xdiff) 
end
function coefficient_of_variation(S::SpikeTrains{R,N},i::I;
    t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where {R,N,I<:Integer}
  train_i = S.trains[i]
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  return coefficient_of_variation(train_i;t_end=t_end,t_start=t_start)
end


function running_covariance_zero_lag(X::Vector{R},Y::Vector{R},Δt::Real,Δt_mean::Real;
        t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where R
  @assert Δt_mean > Δt
  t_end = something(t_end, max(X[end],Y[end])- 0.5*Δt)
  t_start = something(t_start,max(0.0,min(X[1],Y[1]) - 0.5*Δt))
  times,binX = bin_spikes(X,Δt,t_end;t_start=t_start)
  _,binY = bin_spikes(Y,Δt,t_end;t_start=t_start)
  n_mean = round(Integer,Δt_mean/Δt)
  ret = rollcov(binX,binY,n_mean)
  #rateX,rateY = sum.((binX,binY)) ./ (t_end-t_start)
  #ret_noisy = (@. (binX/Δt - rateX)*(binY/Δt-rateY))
  #ret = rollmean(ret_noisy,n_mean)
  return times[n_mean:end],ret
end

function running_covariance_zero_lag(S::SpikeTrains{R,N},ij::Tuple{I,I},Δt::R;
      t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where {R,N,I<:Integer}
  train_i = S.trains[ij[1]]
  train_j = S.trains[ij[2]]
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  return running_covariance_zero_lag(train_i,train_j,Δt;t_end=t_end,t_start=t_start)
end


function pearson_correlation(X::Vector{R},Y::Vector{R},dt::R;
        t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing,
        warn_empty::Bool=false) where {R}
  # if empty, return 0.0
  if isempty(X) || isempty(Y)
    if warn_empty
      @warn "One of the spike trains is empty! Correlation set to 0.0"
    end
    return 0.0
  end
  t_end = something(t_end, max(X[end],Y[end])- dt)
  t_start = something(t_start,max(0.0,min(X[1],Y[1]) - dt))
  @assert t_start < t_end "t_start must be smaller than t_end"
  binnedx = bin_spikes(X,dt,t_end;t_start=t_start)[2]
  binnedy = bin_spikes(Y,dt,t_end;t_start=t_start)[2]
  return cor(binnedx,binnedy)
end

function pearson_correlation(trains::SpikeTrains{R,N},ij::Tuple{I,I},dt::Real;
        t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where {R,N,I<:Integer}
  t_start = something(t_start,trains.t_start)
  t_end = something(t_end,trains.t_end)
  return  pearson_correlation(trains.trains[ij[1]],trains.trains[ij[2]],dt;
        t_end=t_end,t_start=t_start)
end

# for the whole population
function pearson_correlation(trains::SpikeTrains{R,N},dt::Real;
    t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing,
    nan_diag::Bool=true) where {R,N}
  n = trains.n_units
  ret = Matrix{R}(undef,n,n)
  for i in 1:n
    for j in 1:i-1
      _ret = pearson_correlation(trains,(i,j),dt;
      t_end=t_end,t_start=t_start)
      ret[i,j] = _ret
      ret[j,i] = _ret
    end
  end
  if nan_diag
    ret[diagind(ret)] .= NaN
  else
    ret[diagind(ret)] .= 1.0
  end
  return ret
end


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



# Generate random spike trains for testing

function make_poisson_samples(rate::R,t_tot::R) where R
  ret = Vector{R}(undef,round(Integer,1.2*rate*t_tot+10)) # preallocate
  t_curr = zero(R)
  k_curr = 1
  while t_curr <= t_tot
    Δt = -log(rand())/rate
    t_curr += Δt
    ret[k_curr] = t_curr
    k_curr += 1
  end
  return keepat!(ret,1:k_curr-2)
end

function make_random_spiketrains(rates::Vector{R},duration::R;t_start=zero(R)) where R 
  trains = make_poisson_samples.(rates,duration)
  if !iszero(t_start)
    foreach(tr->(tr.+=t_start;nothing),trains)
  end
  return SpikeTrains(length(rates),trains,t_start,t_start+duration)
end


mutable struct DiscreteSpikeTrains{R,N}
  n_units::N
  trains::BitArray{2} # trains[i,j] true if neuron i spikes at time j
  t_start::R
  t_end::R
  dt::R
end

function DiscreteSpikeTrains(trains::BitArray{2},dt::R;t_start=0.0,t_end=-1.0) where R
  n_units = size(trains,1)
  if t_end < 0.0
    t_end = dt*size(trains,2)
  end
  @assert t_end > t_start "t_end must be larger than t_start"
  @assert t_end >= dt*size(trains,2) "t_end must be consistent!"
  return DiscreteSpikeTrains(n_units,trains,t_start,t_end,dt)
end


function SpikeTrains(discrete::DiscreteSpikeTrains{R,N}) where {R,N}
  n_units = discrete.n_units
  trains = Vector{Vector{R}}(undef,n_units)
  dt = discrete.dt
  t_offset = discrete.t_start - 0.5*dt
  for neu in 1:n_units
    trains[neu] = map( x->t_offset+dt*x,findall(discrete.trains[neu,:]))
  end
  return SpikeTrains(n_units,trains,discrete.t_start,discrete.t_end)
end




end # of module
