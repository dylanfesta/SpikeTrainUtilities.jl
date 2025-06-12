
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
  n = length(rates)
  @assert n > 0 "Rates vector must not be empty"
  neus = collect(1:n) 
  trains = make_poisson_samples.(rates,duration)
  if !iszero(t_start)
    foreach(tr->(tr.+=t_start;nothing),trains)
  end
  return SpikeTrains(n,neus,trains,t_start,t_start+duration)
end



function circular_shift_event_times_overflowcounter(event_times::Vector{R}, shift_amount::R, t_start::R, duration::R) where R
  if isempty(event_times)
    return R[] # Return empty vector if no events
  end
  ret = Vector{R}(undef, length(event_times))
  overflow_counter = 0 # Counter for overflow
  @inbounds for k in eachindex(event_times)
    # normalize event time to [0, duration) relative to t_start
    _retk = event_times[k] - t_start
    # add the shift
    _retk += shift_amount
    # check if wrap around is needed
    if _retk > duration
      overflow_counter += 1 # Increment overflow counter
      _retk -= duration # Wrap around
    end
    # convert back to absolute time in [t_start, t_start + duration)
    _retk += t_start
    # assign
    ret[k] = _retk 
  end
  # sort based on overflow counter using circular shift
  if overflow_counter > 0
    circshift!(ret, overflow_counter)
  end
  return ret, overflow_counter
end


"""
  function circular_shift_of_spiketrains(spk::SpikeTrains)
  
Perform a circular shuffle of the spike trains in `spk`.
This function randomly shifts each spike train by a random amount, wrapping around the duration of the spike trains.
The function returns a new `SpikeTrains` object with the shuffled spike times, and the random shifts applied to each train.
The random shifts are generated uniformly in the range `[0, duration)`, where `duration` is the total duration of the spike trains.

# Arguments
- `spk::SpikeTrains`: The input spike trains to be shuffled. It must have a positive duration.
# Returns
- New `SpikeTrains` object with the shuffled spike times.
- Vector of random shifts applied to each spike train.
"""
function circular_shift_of_spiketrains(spk::SpikeTrains{R, N_units_type, T_units_type}) where {R, N_units_type, T_units_type}
  _duration = duration(spk)

  @assert _duration > zero(R) "Duration must be positive!"

  n_units = spk.n_units
  
  # generate random shifts
  random_shifts = _duration .* rand(n_units) 
  new_trains_content = Vector{Vector{R}}(undef, n_units)

  for (k,theshift) in enumerate(random_shifts)
    original_train = spk.trains[k]
    new_trains_content[k]= circular_shift_event_times_overflowcounter(
        original_train, theshift, spk.t_start, _duration)[1]
  end

  return SpikeTrains(new_trains_content; 
    units=spk.units, t_start=spk.t_start, t_end=spk.t_end) ,
    random_shifts #include shift amount, just in case
end



# TO-DO : test, improve... help!

"""
  function circular_shift_of_spiketrains(sq::SpikeQuantity; shift_quantity::Bool=true)
  
Perform a circular shuffle of the event times in `sq`.
This function randomly shifts each event time series by a random amount, wrapping around the duration of the `SpikeQuantity` object.
If `shift_quantity` is true (default), the corresponding `ys` values are shifted along with their event times, maintaining their association after sorting.
If `shift_quantity` is false, `event_times` are shifted and sorted, but the `ys` vectors for each unit are copied from the original `SpikeQuantity` without reordering. 
Thus, the i-th value in `new_ys[k]` will correspond to the i-th time in the *newly sorted* `new_event_times[k]`, which may differ from its original association if the circular shift altered the temporal order of events.

The function returns a new `SpikeQuantity` object with the shuffled event times (and potentially ys), and the random shifts applied to each series.
The random shifts are generated uniformly in the range `[0, duration)`, where `duration` is the total duration of the `SpikeQuantity`.

# Arguments
- `sq::SpikeQuantity`: The input spike quantity to be shuffled. It must have a positive duration.
- `shift_quantity::Bool`: If true (default), `ys` values follow their corresponding event times through the shift and sort. If false, `ys` vectors are copied directly from the input `sq` for each unit without reordering relative to the new sorted event times.

# Returns
- New `SpikeQuantity` object with the shuffled event times and ys.
- Vector of random shifts applied to each event time series.
"""
function circular_shift_of_spiketrains(sq::SpikeQuantity{R, N_SQ, T_SQ}; 
                                     shift_the_quantity::Bool=true) where {R, N_SQ, T_SQ}
  _duration = duration(sq)

  @assert _duration > zero(R) "Duration must be positive!"

  n_units = sq.n_units
  
  # generate random shifts
  random_shifts = _duration .* rand(R, n_units) 

  new_event_times_content = Vector{Vector{R}}(undef, n_units)
  new_ys_content = Vector{Vector{R}}(undef, n_units)

  for k in 1:n_units
    original_event_times_k = sq.event_times[k]
    original_ys_k = sq.ys[k]

    if isempty(original_event_times_k)
      new_event_times_content[k] = R[]
      new_ys_content[k] = R[]
    else
      shift_amount_k = random_shifts[k]
      if shift_the_quantity
        shifted_trains, overflow_counter = circular_shift_event_times_overflowcounter(
              original_event_times_k, shift_amount_k, sq.t_start, _duration)
        new_event_times_content[k] = shifted_trains
        # sort ys according to the new event times
        new_ys_content[k] = circshift(original_ys_k, overflow_counter)
      else # shift_the_quantity is false
        new_event_times_content[k] = circular_shift_event_times_overflowcounter(
              original_event_times_k, shift_amount_k, sq.t_start, _duration)[1]
        new_ys_content[k] = copy(original_ys_k) 
      end
    end
  end
  new_sq = SpikeQuantity(sq.n_units, sq.units, new_event_times_content, new_ys_content, 
                           sq.t_start, sq.t_end, sq.quantity)
                           
  return new_sq, random_shifts
end