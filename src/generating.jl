
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
