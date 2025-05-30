
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
    bin_spikes(Y::Vector{R},binvector::Vector{R}) where R
# Arguments
  + `Y::Vector{<:Real}` : vector of spike times
  + `binvector::AbstractVector{<:Real}` : vector of bin edges
# Returns
  + `times_bin::Vector{R}` : `times_bin[k]` is the midpoint of the timebin `k` (i.e. `binvector[k] + (binvector[k+1]-binvector[k])/2`)   
  + `binned_spikes::Vector{<:Integer}` : `binned_spikes[k]` is the number of spikes that occur 
      in the timebin `k`  (i.e. between `binvector[k]` and `binvector[k+1]`)
"""
function bin_spikes(Y::Vector{R},binvector::AbstractVector{R}) where R<:Real
  # assert is sorted 
  @assert issorted(binvector) "binvector must be sorted!"
  ret = fill(0,length(binvector)-1)
  for y in Y
    if binvector[1] < y <= last(binvector)
      k = searchsortedfirst(binvector,y)-1
      ret[k] += 1
    end
  end
  return midpoints(binvector),ret
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
  thebins = range(t_start,t_end;step=dt) |> collect
  return bin_spikes(Y,thebins)
  # ret = fill(0,length(times)-1)
  # for y in Y
  #   if t_start < y <= last(times)
  #     k = searchsortedfirst(times,y)-1
  #     ret[k] += 1
  #   end
  # end
  # return midpoints(times),ret
end


"""
    bin_spikes(S::SpikeTrains{R,N},dt::R) where {R,N}
# Arguments
  + S : SpikeTrains object
  + dt : time bin size
# Returns
  + times_bin::Vector{R} : `times_bin[k]` is the midpoint of the timebin `k` (i.e. `S.t_start + (k-1/2)*dt`)   
  + `binned_spikes::Matrix{<:Integer}` : `binned_spikes[k,i]` is the number of spikes that occur 
      in the timebin `k` for neuron `i`  (i.e. between `S.t_start + (k-1)*dt` and `S.t_start + k*dt`)
"""
function bin_spikes(S::SpikeTrains{R,N},dt::R) where {R,N}
  nbins = Int64(fld(duration(S)+eps(10.0),dt))
  times = range(S.t_start,S.t_end;length=nbins+1)
  timesmid = midpoints(times)
  binned = fill(0,(nbins,S.n_units))
  for (neu,train) in enumerate(S.trains)
    for y in train
      if S.t_start < y <= last(times)
        k = searchsortedfirst(times,y)-1
        binned[k,neu] += 1
      end
    end
  end
  return timesmid,binned
end

"""
    bin_spikes(S::SpikeTrains{R,N},binvector::Vector{R}) where {R,N}
# Arguments
  + S : SpikeTrains object
  + binvector : vector of bin edges
# Returns
  + times_bin::Vector{R} : `times_bin[k]` is the midpoint of the timebin `k` (i.e. `binvector[k] + (binvector[k+1]-binvector[k])/2`)   
  + `binned_spikes::Matrix{<:Integer}` : `binned_spikes[k,i]` is the number of spikes that occur 
      in the timebin `k` for neuron `i`  (i.e. between `binvector[k]` and `binvector[k+1]`)
"""
function bin_spikes(S::SpikeTrains{R,N},binvector::Vector{R}) where {R,N}
  # assert is sorted 
  @assert issorted(binvector) "binvector must be sorted!"
  ret = fill(0,(length(binvector)-1,S.n_units))
  for (neu,train) in enumerate(S.trains)
    for y in train
      if binvector[1] < y <= last(binvector)
        k = searchsortedfirst(binvector,y)-1
        ret[k,neu] += 1
      end
    end
  end
  return midpoints(binvector),ret
end


function instantaneous_rates(S::SpikeTrains{R,N},dt::R) where {R,N}
  times,binned = bin_spikes(S,dt)
  return times,binned ./ dt
end

function instantaneous_rates(S::SpikeTrains{R,N},binvector::Vector{R}) where {R,N}
  times,binned = bin_spikes(S,binvector)
  deltats = diff(binvector)
  return times,(binned ./ deltats)
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

function covariance_spike_count(X::Vector{R},Y::Vector{R},Δt::R;
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
  binnedy = bin_spikes(Y,Δt,t_end;t_start=t_start)[2]
  binnedx_norm = binnedx .- mean(binnedx)
  binnedy_norm = binnedy .- mean(binnedy)
  return dot(binnedx_norm,binnedy_norm)/(length(binnedx)-1)
end

function covariance_spike_count(S::SpikeTrains{R,N},i::I,j::I,Δt::R;
      t_end::Union{R,Nothing}=nothing,t_start::Union{R,Nothing}=nothing) where {R,N,I<:Integer}
  train_i = S.trains[i]
  train_j = S.trains[j]
  t_start = something(t_start,S.t_start)
  t_end = something(t_end,S.t_end)
  return covariance_spike_count(train_i,train_j,Δt;t_end=t_end,t_start=t_start)
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
