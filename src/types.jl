
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
  _last(v) = ifelse(isempty(v),zero(R),last(v))
  _t_start = something(t_start,zero(R))
  _t_end = something(t_end, maximum(_last,trains)+ 10*eps(R))
  # let's trim to t_end (also a good idea to copy the vectors!)
  _new_trains = map(tr -> !isnothing(t_end) ? filter(<(t_end), tr) : copy(tr), trains) # copy the trains
  n = length(trains)
  return SpikeTrains(n,_new_trains,_t_start,_t_end)
end


# constructor from spiketimes and spikeneurons
function SpikeTrains(spiketimes::Vector{R},spikeneurons::Vector{I},
    nneus::Integer;t_start::Union{R,Nothing}=nothing,t_end::Union{R,Nothing}=nothing) where {R,I<:Integer}
  _t_start = something(t_start,spiketimes[1] - 10*eps(R))
  _t_end = something(t_end,spiketimes[end] + 10*eps(R))
  trains = Vector{Vector{R}}(undef,nneus)
  for neu in 1:nneus
    idxspike = spikeneurons .== neu
    if any(idxspike)
      trains[neu] = spiketimes[idxspike]
    else
      trains[neu] = Vector{R}()
    end 
  end
  # trim in place based on t_end, unoptimized
  if !isnothing(t_end)
    for _train in trains
      idx_delete = findall(>=(t_end),_train)
      deleteat!(_train,idx_delete)
    end
  end
  return SpikeTrains(nneus,trains,_t_start,_t_end)
end

function duration(S::SpikeTrains{R,N}) where {R,N}
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


