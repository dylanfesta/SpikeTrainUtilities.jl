
"""
  SpikeTrains(n_units::N,trains::Vector{Vector{R}},t_start::R,t_end::R) where {R,N}

Constructs a SpikeTrains object.  
"""
mutable struct SpikeTrains{R,N,T}
  n_units::N
  units::Vector{T} # this is to index the units
  trains::Vector{Vector{R}}
  t_start::R
  t_end::R
end 

function SpikeTrains(trains::Vector{Vector{R}};
      units::Union{Vector{T},Nothing}=nothing,
      t_start::Union{R,Nothing}=nothing,t_end::Union{R,Nothing}=nothing) where {R,T}
  _last(v) = ifelse(isempty(v),zero(R),last(v))
  _t_start = something(t_start,zero(R))
  _t_end = something(t_end, maximum(_last,trains)+ 10*eps(R))
  # let's trim (also a good idea to copy the vectors!)
  _new_trains = map(tr -> filter(_t -> _t_start <= _t <= _t_end, tr) , trains) # copy the trains
  # remove empty trains, and if units are present, remove the corresponding units
  idx_del = findall(isempty, _new_trains)
  deleteat!(_new_trains, idx_del)
  n = length(_new_trains)
  if isnothing(units)
    _units = collect(1:n)
  else
    _units = deepcopy(units)
    deleteat!(_units, idx_del) # remove the corresponding units
    @assert length(_units) == n "Number of units must match n_units"
  end
  return SpikeTrains(n, _units, _new_trains, _t_start, _t_end)
end

# constructor from spiketimes and spikeneurons
function SpikeTrains(spiketimes::Vector{R},spikeneurons::Vector{I};
    n_units::Union{Integer,Nothing}=nothing, 
    t_start::Union{R,Nothing}=nothing, t_end::Union{R,Nothing}=nothing) where {R,I<:Integer}

  @assert !isempty(spiketimes) "spiketimes cannot be empty"
  # The 10*eps adjustment is to ensure derived intervals are inclusive of min/max spikes.
  (min_st,max_st) = extrema(spiketimes)
  @assert all(isfinite, [min_st, max_st]) "spiketimes must be finite numbers"
  eff_t_start = something(t_start, min_st - 10*eps(R))
  eff_t_end = something(t_end, max_st + 10*eps(R))
  @assert eff_t_end >= eff_t_start "t_end must be larger than t_start"
  # check if spiketimes need to be trimmed
  if eff_t_start > min_st || eff_t_end < max_st
    # Trim
    idx_trim = findall(t -> eff_t_start <= t <= eff_t_end, spiketimes)
    spiketimes = spiketimes[idx_trim]
    spikeneurons = spikeneurons[idx_trim]
    @assert !isempty(spiketimes) "spiketimes cannot be empty after trimming"
  end

  # Generate units vector from actual neuron IDs reported in spikeneurons
  if isnothing(n_units)
    the_units = sort(unique(spikeneurons))
    _n_units = length(the_units)
  else
    _n_units = n_units
    the_units = collect(1:_n_units)
  end

  # Create trains corresponding to actual_units
  the_trains = Vector{Vector{R}}(undef, _n_units)
  for i in 1:_n_units
    the_trains[i] = R[] # Initialize with empty spike lists
  end

  # Populate trains, filtering by time.
  # Create a map from unit ID to index in the_trains for efficient lookup.
  unit_to_idx_map = Dict{I, Int}()
  for (i, unit_id) in enumerate(the_units)
    unit_to_idx_map[unit_id] = i
  end

  for k in eachindex(spiketimes)
    neuron_id = spikeneurons[k]
    spike_time = spiketimes[k]
    idx = get(unit_to_idx_map, neuron_id, 0) # Get index for this neuron_id
    @assert idx > 0 "Neuron ID $neuron_id not found in actual units"
    push!(the_trains[idx], spike_time) # Spike times are added in order, so trains remain sorted if spiketimes is sorted.
  end
  return SpikeTrains(n_actual_units, the_units, the_trains, eff_t_start, eff_t_end)
end


# when neuron number is provided, it assumes spikeneurons are 1:n_neurons,
# and uses empty vector for silent neurons
function SpikeTrains(spiketimes::Vector{R},spikeneurons::Vector{I},
    nneus::Integer; t_start::Union{R,Nothing}=nothing, t_end::Union{R,Nothing}=nothing) where {R,I<:Integer}
  @warn "SpikeTrains constructor with `nneus` is deprecated, use SpikeTrains(spiketimes, spikeneurons) instead." 
  return SpikeTrains(spiketimes, spikeneurons;t_start=t_start, t_end=t_end)
end
  

function duration(S::SpikeTrains{R,N,T}) where {R,N,T}
  return S.t_end - S.t_start
end


function Base.cat(S::SpikeTrains{R,N,T}...;dims::Int=1) where {R,N,T}
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
  # check that units are consistent and with same oreder
  the_units = S[1].units
  for s in S[2:end]
    if !all( s.units .== the_units )
      error("SpikeTrains must have same units")
    end
  end
  # okay, now cat
  t_start = S[1].t_start
  t_end = S[end].t_end
  trains = Vector{Vector{R}}(undef,n_units)
  for i in 1:n_units
    trains[i] = cat([s.trains[i] for s in S]...,dims=1)
  end
  return  SpikeTrains(n_units,the_units,trains,t_start,t_end)
end

@inline function Base.minimum(S::SpikeTrains{R,N,T}) where {R,N,T}
  _first(v) = ifelse(isempty(v),Inf,first(v))
  minimum(_first, S.trains)
end
@inline function Base.maximum(S::SpikeTrains{R,N,T}) where {R,N,T}
  _last(v) = ifelse(isempty(v),-Inf,last(v))
  maximum(last, S.trains)
end


# This is to merge them together, horizontally, so to speak
# TO-DO ... how to deal with same units????
function Base.merge(S::SpikeTrains{R,N,T}...) where {R,N,T}
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

function duration(S::DiscreteSpikeTrains{R,N}) where {R,N}
  return S.t_end - S.t_start
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


# Spike quantity is for quantities that I can associate to the discrete spike events
# mostly instantaneous firing rates, defined as 1/ISI

struct SpikeQuantity{R,N,T}
  n_units::N
  units::Vector{T} # this is to index the units, T can be any type
  event_times::Vector{Vector{R}}
  ys::Vector{Vector{R}}
  t_start::R
  t_end::R
  quantity::Symbol # e.g. :spikes, :isi, :psth, etc.
end

function duration(S::SpikeQuantity{R,N,T}) where {R,N,T}
  return S.t_end - S.t_start
end

function SpikeQuantity(n_units::N,event_times::Vector{Vector{R}},
    ys::Vector{Vector{R}}; 
    units::Union{Vector{T},Nothing}=nothing,
    t_start::Union{R,Nothing}=nothing,t_end::Union{R,Nothing}=nothing,
    quantity::Symbol=:spikes) where {R,N,T}
  _t_start = something(t_start,minimum(event_times))
  _t_end = something(t_end,maximum(event_times))
  @assert length(event_times) == n_units "Number of event times must match n_units"
  @assert length(ys) == n_units "Number of ys must match n_units"
  @assert all(length.(event_times) .== length.(ys)) "Each unit must have the same number of events and ys"
  if isnothing(units)
    _units = collect(1:n_units)
  else
    _units = deepcopy(units)
    @assert length(_units) == n_units "Number of units must match n_units"
  end
  return SpikeQuantity(n_units,_units,event_times,ys,_t_start,_t_end,quantity)
end

struct BinnedSpikeQuantity{R,RT,N,T}
  n_units::N
  n_bins::N
  ys::Matrix{R} # ys[i,j] is the value for unit i at bin j, therefore size(ys) == (n_units,n_bins)
  dt::RT # bin width
  units::Vector{T} # this is to index the units, T can be any type
  t_start::RT
  t_end::RT
  quantity::Symbol # e.g. :spikes, :isi, :psth, etc.
end

function duration(S::BinnedSpikeQuantity{R,N,T}) where {R,N,T}
  return S.t_end - S.t_start
end

function check_time_consistency(S::BinnedSpikeQuantity{R,N,T}) where {R,N,T}
  @assert S.t_start <= S.t_end "t_start must be less than or equal to t_end"
  @assert S.n_bins > 0 "n_bins must be greater than zero"
  @assert S.dt > 0.0 "dt must be greater than zero"
  @assert isapprox(duration(S),S.n_bins * S.dt) "t_end must match t_start + n_bins * dt"
  return nothing
end


function get_t_midpoints(S::BinnedSpikeQuantity{R,N,T}) where {R,N,T}
  return collect(range(S.t_start + 0.5*S.dt; length=S.n_bins, step=S.dt))
end

function get_t_edges(S::BinnedSpikeQuantity{R,N,T}) where {R,N,T}
  return collect(range(S.t_start; length=S.n_bins+1, step=S.dt))
end

function get_t_edges(trs::Union{SpikeTrains{R,N,T},SpikeQuantity{R,N,T}},dt::R) where {R,N,T}
  t_start = max(0.0,trs.t_start-10.0*eps(R)) # in case it's the first spike
  Δt = duration(trs)
  n_bins = Int64(fld(Δt+eps(Δt), dt)) + 1  # number of bins
  return collect(range(t_start; length=n_bins, step=dt))
end

function get_t_midpoints(trs::Union{SpikeTrains{R,N,T},SpikeQuantity{R,N,T}},dt::R) where {R,N,T}
  t_edges = get_t_edges(trs,dt)
  return (t_edges[1:end-1] .+ t_edges[2:end]) ./ 2.0
end


# now, the binning methods, for now I keep the dt outside... is this the best way?
abstract type AbstractBinning end

# This is a binning method that sums the values in each bin
struct BinSum <: AbstractBinning end

# This is a binning method that averages the values in each bin
struct BinMean <: AbstractBinning  end

# This is a binning method that counts the number of events in each bin
struct BinCount <: AbstractBinning end

# This is the number of events divided by the bin width
struct BinRate <: AbstractBinning end

# This is a binning method that takes the maximum value in each bin
struct BinMaxPooling <: AbstractBinning end

struct BinGaussianKernel <: AbstractBinning
  σ::Real # standard deviation of the Gaussian kernel
end

struct BinCausalGaussianKernel <: AbstractBinning
  σ::Real # standard deviation of the Gaussian kernel
end