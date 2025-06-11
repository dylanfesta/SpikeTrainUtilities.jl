

function _filter_small_isis!(spiketimes::AbstractVector{R}; threshold::Real=0.004) where R
  isis = diff(spiketimes)
  n_isis = length(isis)
  to_remove = Int64[]
  @inbounds for k in 1:n_isis-1
    isis_val = isis[k]
    if isis_val < threshold
      push!(to_remove, k+1)
      isis[k+1] += isis_val # correct the next ISI
    end
  end
  if isis[n_isis] < threshold
    push!(to_remove, n_isis+1)
  end
  deleteat!(spiketimes, to_remove)
  return nothing # returns nothing, because such is life
end


"""
  function filter_out_small_isi!(spiketrains::Vector{<:AbstractVector{<:Real}}, dt_min::Real)

Filter out spikes (events) that are closer than `dt_min` from each other in the given spiketrains.
This modifies the input `spiketrains` in place, removing spikes that are too close together.

# Arguments:
- `spiketrains::Vector{<:AbstractVector{<:Real}}`: A vector of spike trains, where each train is an array of spike times.
- `dt_min::Real`: The minimum time difference between spikes. Spikes closer than this will be removed.
# Returns:
- nothing, modifies `spiketrains` in place.
"""
function filter_out_small_isi!(spiketrains::Vector{<:AbstractVector{<:Real}}, dt_min::Real)
  for i in eachindex(spiketrains)
    spiketimes = spiketrains[i]
    if length(spiketimes) > 1
      _filter_small_isis!(spiketimes; threshold=dt_min)
    end
  end
  return nothing
end


"""
  function filter_out_small_isi!(spiketrains::SpikeTrains, dt_min::Real)

Filter out spikes (events) that are closer than `dt_min` from each other in the spike trains object
This modifies the input `spiketrains` in place, removing spikes that are too close together.
# Arguments:
- `spiketrains::SpikeTrains`: An instance of `SpikeTrains`, which contains multiple spike trains.
- `dt_min::Real`: The minimum time difference between spikes. Spikes closer than this will be removed.
# Returns:
- nothing, modifies `spiketrains` in place.
"""
function filter_out_small_isi!(spiketrains::SpikeTrains, dt_min::Real)
  filter_out_small_isi!(spiketrains.trains,dt_min)
  return nothing
end


"""
  function get_instantaneous_firing_rates(spiketrains::SpikeTrains; dt_min::Real=0.0)

Calculate the instantaneous firing rates, iFR, defined as 1/ISI, for each spike train in the `spiketrains` object.
The iFR time is defined as the time of the right spike in the ISI.

# Arguments:
- `spiketrains::SpikeTrains`: An instance of `SpikeTrains`, which contains multiple spike trains.
- `dt_min::Real`: The minimum time difference between spikes. Spikes closer than this will be removed before calculating iFR,
  default is 0.0, which means no filtering.
# Returns:
- `iFRTrains::SpikeQuantity`: An instance of `SpikeQuantity` containing the instantaneous firing rates for each spike train.
  The `event_times` are the times of the left spikes in the ISI, and `ys` are the corresponding iFR values.
"""
function get_instantaneous_firing_rates(spiketrains::SpikeTrains{R,N,T}; dt_min::Real=0.0) where {R,N,T}

  # Determine the trains to process
  # If dt_min is specified, filter a copy of the trains
  trains_to_process = if dt_min > 0.0
    temp_trains = deepcopy(spiketrains.trains)
    filter_out_small_isi!(temp_trains, dt_min)
    temp_trains
  else
    spiketrains.trains
  end

  n_units = spiketrains.n_units

  all_event_times = Vector{Vector{R}}(undef, n_units)
  all_ifrs = Vector{Vector{R}}(undef, n_units)

  for i in 1:n_units
    current_train = trains_to_process[i] 

    if length(current_train) < 2
      all_event_times[i] = R[]
      all_ifrs[i] = R[]
    else
      all_event_times[i] = current_train[2:end] # Event times are the right spike in the ISI
      isis = diff(current_train) # Vector of ISIs
      all_ifrs[i] = inv.(isis)
    end
  end

  return SpikeQuantity(n_units, all_event_times, all_ifrs;
                       units=spiketrains.units,
                       t_start=spiketrains.t_start, 
                       t_end=spiketrains.t_end, 
                       quantity=:iRF)
end



## Discretize functions!

# error fallback

function discretize(spkq::Union{SpikeQuantity,SpikeTrains}, dt::Real,::AbstractBinning)
  error("Discretization method not implemented!!!")
end


# utility function
function _count_between_edges(times::AbstractVector{<:Real},edges::AbstractVector{<:Real})
  nret = length(edges)-1
  ret = zeros(Int64,nret)
  for t in times 
    bin_idx = searchsortedfirst(edges,t) - 1
    if bin_idx > 0 && bin_idx <= nret
      ret[bin_idx] += 1
    end
  end
  return ret
end

function _sum_between_edges(times::AbstractVector{<:Real},x::Vector{R1},edges::AbstractVector{<:Real}) where R1<:Real
  @assert length(x) == length(times) "Length of x and times do not match!"
  nret = length(edges)-1
  ret = zeros(R1,nret)
  for (t,xval) in zip(times,x) 
    bin_idx = searchsortedfirst(edges,t) - 1
    if bin_idx > 0 && bin_idx <= nret
      ret[bin_idx] += xval
    end
  end
  return ret
end

function _average_between_edges(times::AbstractVector{<:Real},x::Vector{R1},edges::AbstractVector{<:Real}) where R1<:Real
  @assert length(x) == length(times) "Length of x and times do not match!"
  nret = length(edges)-1
  counts = zeros(Int64,nret)
  ret = zeros(R1,nret)
  for (t,xval) in zip(times,x) 
    bin_idx = searchsortedfirst(edges,t) - 1
    if bin_idx > 0 && bin_idx <= nret
      ret[bin_idx] += xval
      counts[bin_idx] += 1
    end
  end
  @inbounds for k in 1:nret
    if counts[k] > 0
      ret[k] /= counts[k]
    end
  end
  return ret
end

function _max_pooling_between_edges(times::AbstractVector{<:Real},x::Vector{RX},edges::AbstractVector{<:Real}) where RX
  @assert length(x) == length(times) "Length of x and times do not match!"
  nret = length(edges)-1
  ret = zeros(RX,nret)
  for (t,xval) in zip(times,x) 
    bin_idx = searchsortedfirst(edges,t) - 1
    if bin_idx > 0 && bin_idx <= nret
      ret[bin_idx] = max(ret[bin_idx],xval)
    end
  end
  return ret
end


function discretize(spkq::SpikeQuantity{R,N,T}, dt::Real,::BinSum) where {R,N,T}
  n_units = spkq.n_units
  t_edges = get_t_edges(spkq,dt)
  new_t_end=t_edges[end]
  n_bins = length(edges) - 1
  ret_y = Matrix{R}(undef, n_units,n_bins)
  for i in 1:n_units
    ret_y[i,:] = _sum_between_edges(spkq.event_times[i], spkq.ys[i], t_edges)
  end
  ret = BinnedSpikeQuantity(n_units, n_bins,ret_y, dt,
    spkq.units,spkq.t_start, new_t_end,:binned_sum)
  check_time_consistency(ret)
  return ret
end

function discretize(spkq::SpikeTrains{R,N,T}, dt::Real,::BinCount) where {R,N,T}
  n_units = spkq.n_units
  t_edges = get_t_edges(spkq,dt)
  new_t_end=t_edges[end]
  n_bins = length(t_edges) - 1
  ret_y = Matrix{N}(undef, n_units,n_bins)
  for k in 1:n_units
    ret_y[k,:] = _count_between_edges(spkq.trains[k], t_edges)
  end
  ret = BinnedSpikeQuantity(n_units, n_bins,ret_y, dt,
    spkq.units,spkq.t_start, new_t_end,:binned_spike_count)
  check_time_consistency(ret)
  return ret
end

function discretize(spkq::SpikeQuantity{R,N,T}, dt::Real,::BinMean) where {R,N,T}
  n_units = spkq.n_units
  t_edges = get_t_edges(spkq,dt)
  new_t_end=t_edges[end]
  n_bins = length(t_edges) - 1
  ret_y = Matrix{R}(undef, n_units,n_bins)
  for i in 1:n_units
    ret_y[i,:] = _average_between_edges(spkq.event_times[i], spkq.ys[i], t_edges)
  end
  ret = BinnedSpikeQuantity(n_units, n_bins,ret_y, dt,
    spkq.units,spkq.t_start, new_t_end,:binned_mean)
  check_time_consistency(ret)
  return ret
end



function discretize(spkq::SpikeQuantity{R,N,T}, dt::Real,::BinMaxPooling) where {R,N,T}
  n_units = spkq.n_units
  t_edges = get_t_edges(spkq,dt)
  new_t_end = t_edges[end]
  n_bins = length(t_edges) - 1
  ret_y = Matrix{R}(undef, n_units,n_bins)
  for i in 1:n_units
    ret_y[i,:] = _max_pooling_between_edges(spkq.event_times[i], spkq.ys[i], t_edges)
  end
  ret = BinnedSpikeQuantity(n_units, n_bins,ret_y, dt,
    spkq.units,spkq.t_start, new_t_end,:binned_maxpooling)
  check_time_consistency(ret)
  return ret
end


function discretize(spkq::SpikeTrains{R,N,T}, dt::Real, binning::BinGaussianKernel) where {R,N,T}
  n_units = spkq.n_units
  # Calculate bin centers based on the spike train's time range and dt
  t_centers = get_t_midpoints(spkq, dt)
  n_bins = length(t_centers)
  σ = binning.σ

  # Initialize the result matrix for binned values
  ret_y = zeros(R, n_units, n_bins)

  # Gaussian kernel function (unnormalized)
  # G(t_diff) = exp(-(t_diff^2) / (2 * σ^2))
  g(t_diff, sigma) = exp(-(t_diff^2) / (2 * sigma^2))

  # Define a cutoff for kernel influence (e.g., 4 standard deviations)
  # Spikes outside this range contribute negligibly to a bin
  cutoff_sigma_multiplier = 4.0
  time_cutoff = cutoff_sigma_multiplier * σ

  # Iterate through each unit
  for i in 1:n_units
    current_train = spkq.trains[i]

    # Iterate through each spike in the current train
    for t_spike in current_train
      # Calculate the range of bin centers potentially affected by this spike
      # A bin center k is located at t_start + (k-1 + 0.5) * dt
      # We are interested in bins where |t_spike - t_center_k| <= time_cutoff
      # t_spike - time_cutoff <= t_start + (k-1 + 0.5) * dt <= t_spike + time_cutoff
      # Rearranging for k-1:
      # (t_spike - time_cutoff - t_start - 0.5*dt) / dt <= k-1 <= (t_spike + time_cutoff - t_start - 0.5*dt) / dt
      # Rearranging for k:
      # (t_spike - time_cutoff - t_start - 0.5*dt) / dt + 1 <= k <= (t_spike + time_cutoff - t_start - 0.5*dt) / dt + 1

      # Calculate the approximate float indices for the start and end of the relevant bin range
      k_float_start = (t_spike - time_cutoff - spkq.t_start - 0.5*dt) / dt + 1
      k_float_end = (t_spike + time_cutoff - spkq.t_start - 0.5*dt) / dt + 1

      # Convert float indices to integer indices, clamping to the valid bin range [1, n_bins]
      k_start = max(1, floor(Int, k_float_start))
      k_end = min(n_bins, ceil(Int, k_float_end))

      # Iterate over the relevant bins and add the spike's contribution
      for k in k_start:k_end
        t_center = t_centers[k] # t_centers is 1-indexed
        t_diff = t_spike - t_center
        ret_y[i, k] += g(t_diff, σ)
      end
    end
  end
  t_end_new = spkq.t_start + n_bins * dt
  # Construct the BinnedSpikeQuantity object
  ret = BinnedSpikeQuantity(n_units, n_bins, ret_y, dt,
    spkq.units, spkq.t_start, t_end_new, :binned_gaussian_kernel)

  check_time_consistency(ret) # Ensure the resulting object is consistent

  return ret
end



function discretize(spkq::SpikeTrains{R,N,T}, dt::Real, binning::BinCausalGaussianKernel) where {R,N,T}
  n_units = spkq.n_units
  # Calculate bin centers based on the spike train's time range and dt
  t_centers = get_t_midpoints(spkq, dt)
  n_bins = length(t_centers)
  σ = binning.σ

  # Initialize the result matrix for binned values
  ret_y = zeros(R, n_units, n_bins)

  # Causal Gaussian kernel function (unnormalized)
  # G(t_diff) = exp(-(t_diff^2) / (2 * σ^2)) if t_diff <= 0, else 0
  # t_diff here is t_spike - t_center
  g_causal(t_diff, sigma) = t_diff <= 0 ? exp(-(t_diff^2) / (2 * sigma^2)) : zero(R)

  # Define a cutoff for kernel influence (e.g., 4 standard deviations)
  # Spikes outside this range contribute negligibly to a bin.
  # For a causal kernel, a spike at t_spike only affects bins with t_center >= t_spike.
  # The kernel value is significant for t_center within [t_spike, t_spike + time_cutoff].
  cutoff_sigma_multiplier = 4.0
  time_cutoff = cutoff_sigma_multiplier * σ

  # Iterate through each unit
  for i in 1:n_units
    current_train = spkq.trains[i]

    # Iterate through each spike in the current train
    for t_spike in current_train
      # Calculate the range of bin centers potentially affected by this spike
      # We are interested in bins where t_center is in [t_spike, t_spike + time_cutoff]
      # t_center_k = t_start + (k-1 + 0.5) * dt
      # t_spike <= t_start + (k-1 + 0.5) * dt <= t_spike + time_cutoff
      # Rearranging for k-1:
      # (t_spike - t_start - 0.5*dt) / dt <= k-1 <= (t_spike + time_cutoff - t_start - 0.5*dt) / dt
      # Rearranging for k:
      # (t_spike - t_start - 0.5*dt) / dt + 1 <= k <= (t_spike + time_cutoff - t_start - 0.5*dt) / dt + 1

      # Calculate the approximate float indices for the start and end of the relevant bin range
      k_float_start = (t_spike - spkq.t_start - 0.5*dt) / dt + 1 # Start from the bin containing t_spike
      k_float_end = (t_spike + time_cutoff - spkq.t_start - 0.5*dt) / dt + 1

      # Convert float indices to integer indices, clamping to the valid bin range [1, n_bins]
      k_start = max(1, floor(Int, k_float_start))
      k_end = min(n_bins, ceil(Int, k_float_end))

      # Iterate over the relevant bins and add the spike's contribution
      for k in k_start:k_end
        t_center = t_centers[k] # t_centers is 1-indexed
        t_diff = t_spike - t_center # t_diff is negative or zero for causal contribution
        ret_y[i, k] += g_causal(t_diff, σ)
      end
    end
  end
  t_end_new = spkq.t_start + n_bins * dt
  ret = BinnedSpikeQuantity(n_units, n_bins, ret_y, dt, spkq.units, spkq.t_start, t_end_new, :binned_causal_gaussian_kernel)
  check_time_consistency(ret)
  return ret
end