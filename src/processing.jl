

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
function get_instantaneous_firing_rates(spiketrains::SpikeTrains{R,N}; dt_min::Real=0.0) where {R,N}

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

  return SpikeQuantity(n_units, all_event_times, all_ifrs, 
                       t_start=R_quant(spiketrains.t_start), 
                       t_end=R_quant(spiketrains.t_end), 
                       quantity=:iRF)
end
