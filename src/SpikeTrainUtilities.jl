module SpikeTrainUtilities
using LinearAlgebra,Statistics,StatsBase
using Colors
using RollingFunctions

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

## analysis here

"""
  numerical_rates(S::SpikeTrains{R,N};t_start::R=0.0,t_end::R=Inf) where {R,N}

Compute the numerical rates of the spike trains in `S` over the interval
 `[t_start,t_end]` if specified. If not, uses the interval `[S.t_start,S.t_end]`.
"""
function numerical_rates(S::SpikeTrains{R,N};
    t_start::Union{R,Nothing}=nothing,t_end::Union{R,Nothing}=nothing) where {R,N}
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  Δt = t_end - t_start
  return [count( spk ->  t_start<= spk <= t_end ,train)/Δt for train in S.trains]
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
  nbins = div(delta_t(S),dt)
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
  return midpoints(times),ret
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
  times_cov_ret = vcat(-reverse(times_cov[2:end]),times_cov)
  ret = Vector{Float64}(undef,2*ndt-1)
  binnedx = bin_spikes(X,dτ,t_end;t_start=t_start)[2]
  binnedy = bin_spikes(Y,dτ,t_end;t_start=t_start)[2]
  rx = length(X) / (t_end-t_start) # mean rate
  ry = length(Y) / (t_end-t_start) # mean rate
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
  @assert t_start + duration <= S.t_end "t_start + duration is too large"
  trains = S.trains
  return draw_spike_raster(trains,dt,time_duration;
    t_start=t_start,
    spike_size=spike_size,
    spike_separator=spike_separator,
    background_color=background_color,
    spike_colors=spike_colors,
    max_size=max_size)
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




#=

function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R;
   t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing,
   verbose::Bool=false) where R
  t_end = something(t_end, maximum(last,Ys)- dτ)
  t_start = something(t_start,zero(R))
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Array{Float64}(undef,ndt,n,n)
  if verbose
      @info "The full dynamical iteration has $(round(Integer,t_end/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,t_end;t_start=t_start)[2]
    fmi = length(Ys[i]) / (t_end - t_start) 
    ndt_tot = length(binnedi)
    _ret_alloc = Vector{R}(undef,ndt)
    for j in 1:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,t_end;t_start=t_start)[2]
      fmj = length(Ys[j]) / t_end # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        _ret_alloc[k+1] = dot(binnedi,binnedj_sh)
      end
      @. _ret_alloc = (_ret_alloc / (ndt_tot*dτ^2)) - fmi*fmj
      ret[:,i,j] = _ret_alloc
    end
  end
  return get_times_strict(dτ,τmax), ret
end

##########################
## covariance density

@inline function get_times_strict(dt::R,Tend::R;Tstart::R=0.0) where R<:Real
  return range(Tstart,Tend-dt;step=dt)
end


# and returns a value in Hz
function instantaneous_rates(idxs_neu::AbstractVector{<:Integer},
    dt::Float64,pop::AbstractPopulation;Tend::Float64=-1.0,Tstart::Float64=0.0)
  trains = pop.state.trains_history[idxs_neu]  
  return  instantaneous_rates(dt,trains;
   Tend=Tend,Tstart=Tstart)
end
function instantaneous_rates(dt::Float64,trains::Vector{Vector{Float64}};
  Tend::Float64=-1.0,Tstart::Float64=0.0)
  if Tend < 0 
    Tend = minimum(last.(trains))
  end
  tmid = midpoints(range(Tstart,Tend;step=dt))
  counts = fill(0.0,length(tmid))
  for train in trains
    counts .+= bin_spikes(train,dt,Tend;Tstart=Tstart)
  end
  return tmid,counts ./ (dt*length(trains))
end

# time starts at 0, ends at T-dt, there are T/dt steps in total
@inline function get_times(dt::Real,T::Real)
  return (0.0:dt:(T-dt))
end


# the first element (zero lag) is always rate/dτ
function covariance_self_numerical(Y::Vector{R},dτ::R,τmax::R,
     Tend::Union{R,Nothing}=nothing) where R
  τtimes,ret = covariance_density_numerical([Y,],dτ,τmax;verbose=false,Tend=Tend)
  return  τtimes, ret[:,1,1]
end


function covariance_density_numerical(Ys::Vector{Vector{R}},dτ::Real,τmax::R;
   Tend::Union{R,Nothing}=nothing,verbose::Bool=false) where R
  Tend = something(Tend, maximum(last,Ys)- dτ)
  ndt = round(Integer,τmax/dτ)
  n = length(Ys)
  ret = Array{Float64}(undef,ndt,n,n)
  if verbose
      @info "The full dynamical iteration has $(round(Integer,Tend/dτ)) bins ! (too many?)"
  end
  for i in 1:n
    binnedi = bin_spikes(Ys[i],dτ,Tend)
    fmi = length(Ys[i]) / Tend # mean frequency
    ndt_tot = length(binnedi)
    _ret_alloc = Vector{R}(undef,ndt)
    for j in 1:n
      if verbose 
        @info "now computing cov for pair $i,$j"
      end
      binnedj =  i==j ? binnedi : bin_spikes(Ys[j],dτ,Tend)
      fmj = length(Ys[j]) / Tend # mean frequency
      binnedj_sh = similar(binnedj)
      @inbounds @simd for k in 0:ndt-1
        circshift!(binnedj_sh,binnedj,k)
        _ret_alloc[k+1] = dot(binnedi,binnedj_sh)
      end
      @. _ret_alloc = (_ret_alloc / (ndt_tot*dτ^2)) - fmi*fmj
      ret[:,i,j] = _ret_alloc
    end
  end
  return get_times_strict(dτ,τmax), ret
end

function covariance_density_ij(Ys::Vector{Vector{R}},i::Integer,j::Integer,dτ::R,τmax::R;
   Tend::Union{R,Nothing}=nothing) where R
  return covariance_density_ij(Ys[i],Ys[j],dτ,τmax;Tend=Tend)
end

function covariance_density_ij(X::Vector{R},Y::Vector{R},dτ::Real,τmax::R;
    Tend::Union{R,Nothing}=nothing) where R
  times = get_times_strict(dτ,τmax)
  ndt = length(times)
  times_ret = vcat(-reverse(times[2:end]),times)
  Tend = something(Tend, max(X[end],Y[end])- dτ)
  ret = Vector{Float64}(undef,2*ndt-1)
  binnedx = bin_spikes(X,dτ,Tend)
  binnedy = bin_spikes(Y,dτ,Tend)
  fx = length(X) / Tend # mean frequency
  fy = length(Y) / Tend # mean frequency
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
  @. ret = (ret / (ndt_tot*dτ^2)) - fx*fy
  return times_ret, ret
end



# frequencies for Fourier transform.
# from -1/dt to 1/dt - 1/T in steps of 1/T
function get_frequencies_centerzero(dt::Real,T::Real)
  dω = inv(T)
  ωmax = 0.5/dt
  f = dω:dω:ωmax
  ret = vcat(-reverse(f),0.,f[1:end-1])
  return ret
end

function get_frequencies(dt::Real,T::Real)
  dω = inv(T)
  ωmax = inv(dt)
  ret = 0:dω:ωmax-dω
  return ret
end


# do draw spikes on existing raster plot
function add_to_spike_raster!(raster::Matrix,
    trains::Vector{Vector{Float64}},neu_idxs::Vector{Int64},
    dt::Real,Tend::Real, spike_color::C;
      Tstart::Real=0.0,
      spike_size::Integer = 5,
      spike_separator::Integer = 1,
      ) where C<:Color
  @assert length(neu_idxs) == length(trains)    
  binned_binary  = map(trains) do train
    .! iszero.(bin_spikes(train,dt,Tend;Tstart=Tstart))
  end
  ntimes = length(binned_binary[1])
  @assert size(raster,2) == ntimes "Mismatch in time binning!"
  for (neu,binv) in zip(neu_idxs,binned_binary)
    spk_idx = findall(binv)
    _idx_pre = (neu-1)*(spike_size+spike_separator)+spike_separator
    y_color = _idx_pre+1:_idx_pre+spike_size
    raster[y_color,spk_idx] .= spike_color
  end
  return nothing
end


#### this is the part which involves reader objects

function get_spiketimes_spikeneurons(rec::Union{RecFullTrain,RecFullTrainContent},
    (pop_idx::Integer)=1)
  spiketimes,spikeneurons = rec.timesneurons[pop_idx]
  spkt = filter(isfinite,spiketimes)
  spkn = filter(>(0),spikeneurons)
  return spkt,spkn
end
function spiketn_to_trains(spiketimes::Vector{<:Real},
    spikeneurons::Vector{<:Integer},Nneus::Integer)
  trains = map(_-> Float64[],1:Nneus)
  for (spkt,spkn) in zip(spiketimes,spikeneurons)
    push!(trains[spkn],spkt)
  end
  return trains
end
function get_trains(rec::Union{RecFullTrain,RecFullTrainContent},Nneus::Integer,
    (pop_idx::Integer)=1)
  spiketn = get_spiketimes_spikeneurons(rec,pop_idx)
  return spiketn_to_trains(spiketn...,Nneus)
end

"""
  numerical_rates(spikes_recorder,Nneus,Tend;pop_idx=1,Tstart=0.0)

  Returns the numerical rates of the neurons in the population `pop_idx` 
  that has been recorded through `spikes_recorder` (a `RecFullTrain` or a 
  `RecFullTrainContent` object). The rates are computed in the time interval 
  `[Tstart,Tend]` and are returned as a vector of length `Nneus`.
"""
function numerical_rates(rec::Union{RecFullTrain,RecFullTrainContent},Nneus::Integer,Tend::Real;
    pop_idx::Integer=1,Tstart::Real=0.0)
  @assert Tstart < Tend "Tstart must be smaller than Tend, got $Tstart and $Tend instead"  
  trains = get_trains(rec,Nneus,pop_idx)  
  return numerical_rate.(trains;Tstart=Tstart,Tend=Tend)
end

function binned_spikecount(trains::Vector{Vector{R}},dt::Float64,Tend::Real;
      neurons_idx::AbstractVector=Int64[],
      Tstart::Float64=0.0) where R
  if !isempty(neurons_idx)
    trains = trains[neurons_idx]
  end
  Nneus = length(trains)
  tbins = Tstart:dt:Tend
  ntimes = length(tbins)-1
  binnedcount = fill(0,(Nneus,ntimes))
  for (neu,train) in enumerate(trains)
    for t in train
      if (tbins[1] < t <= tbins[end]) # just in case
        tidx = searchsortedfirst(tbins,t)-1
        binnedcount[neu,tidx]+=1
      end
    end
  end
  binsc=midpoints(tbins)
  return binsc,binnedcount
end


function instantaneous_rates(trains::Vector{Vector{R}},dt::R,Tend::Real;
    neurons_idx::AbstractArray=Int64[],
    Tstart::Float64=0.0) where R
  @assert Tend > Tstart+dt "Tend should be greater than Tstart+dt, but got Tend=$Tend and Tstart+dt=$(Tstart+dt)"  
  binsc,counts = binned_spikecount(trains,dt,Tend;neurons_idx=neurons_idx,Tstart=Tstart)  
  return binsc, (counts./dt)
end


=#





















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





end
