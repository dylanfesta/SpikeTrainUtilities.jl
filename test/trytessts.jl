##
push!(LOAD_PATH, abspath(@__DIR__,".."))
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Makie,CairoMakie
using Random
using Distributions
Random.seed!(0);
import StatsBase: midpoints


## 

my_units = collect('a':'z')
n = length(my_units)

the_rates = 60 .* rand(n) 
T = 1000.0
trains = [U.make_poisson_samples(rate,T) for rate in the_rates ]

spiketrains = U.SpikeTrains(trains; t_start=0.0, t_end=T,
  units=my_units)


units_shuffled = shuffle(my_units)
resorting_dict1 = Dict(zip(units_shuffled,1:n))
resorting_dict2 = Dict(zip(units_shuffled,0:n-1))

U.sort_units!(spiketrains,resorting_dict1)


##

n  = 44
T = 1000.0
dt = 0.1
trains = U.make_random_spiketrains(the_rates,T)

trains_shifted = U.circular_shift_of_spiketrains(trains)[1]

# check all trains are different
for neu in 1:n
  t1 = trains.trains[neu]
  t2 = trains_shifted.trains[neu]
  @show issorted(t1),issorted(t2)
  @assert all(t1 .!= t2) "Train $neu is not shifted"
end


##
trains.trains[1][1:30]
trains_shifted.trains[1][1:30]

##

edges_test1 = U.get_t_edges(trains1,dt)
centers_test1= U.get_t_midpoints(trains1,dt)

trains_bincounts = U.discretize(trains1,dt,U.BinCount())
edges_test2 = U.get_t_edges(trains_bincounts)
@assert all(edges_test1 .== edges_test2)
for (k,_train) in enumerate(trains1.trains)
  _train_less = _train[_train .< trains_bincounts.t_end]
  @assert length(_train_less) == sum(trains_bincounts.ys[k,:])
end


##


##
error()
##
n  = 44
rates1 = 250 .* rand(n) 
T = 1000.0
trainstest = let _tr =  U.make_random_spiketrains(rates1,T)
  U.SpikeTrains(_tr.trains)
end

# compute iFR with cutoff at 5ms
iFRs = U.get_instantaneous_firing_rates(trainstest; dt_min=5E-3)


# test they are all below 200 Hz
for y in iFRs.ys
  @assert all(<=(200.0),y)
end

##
iFRs.t_end
dt_test = 0.1
duration = U.duration(iFRs)
t_edges = U.get_t_edges(trainstest,0.1)
n_bins = length(t_edges)-1
n_bins*dt_test

iFRs_maxpool = U.discretize(iFRs,0.1,U.BinMaxPooling())

##

error()

##

n_borders = 33

borders = rand(Uniform(66.0,99.0),n_borders) |> sort

n_spikes1 = [ rand(0:100) for _ in 1:n_borders-1]
n_spikes2 = [ rand(0:100) for _ in 1:n_borders-1]

fake_spikes1 = let ret=[]
  for i in 1:n_borders-1
    push!(ret,rand(Uniform(borders[i],borders[i+1]),n_spikes1[i]))
  end
  vcat(ret...)
end
fake_spikes2 = let ret=[]
  for i in 1:n_borders-1
    push!(ret,rand(Uniform(borders[i],borders[i+1]),n_spikes2[i]))
  end
  vcat(ret...)
end

_,n_spikes1_check=U.bin_spikes(fake_spikes1,borders)

_spiketrain = U.SpikeTrains([fake_spikes1,fake_spikes2];t_start=0.0,t_end=100.0)

_, n_spikes_check = U.bin_spikes(_spiketrain,borders)
@assert all(n_spikes_check[:,1] .== n_spikes1)
@assert all(n_spikes_check[:,2] .== n_spikes2)


mid_p, instrates = U.instantaneous_rates(_spiketrain,borders)

all(isapprox.(mid_p,midpoints(borders)))
all(isapprox.(instrates[:,1],n_spikes1./diff(borders)))


##

rate = 100.0
T = 10_000.0
train1 = U.make_poisson_samples(rate,T)

train2 = let ret = Float64[]
  t_now = 0.0
  t_end = train1[end]
  for tt in train1
    if (tt/t_end>0.5) &&  (rand() <  0.02*tt/t_end)
      t_now = tt + randn()*0.2
    else
      Δt = -log(rand())/rate
      t_now += Δt
    end
    push!(ret,t_now)
  end
  sort(ret)
end

##

U.draw_spike_raster([train1,train2],1E-3,1.0;
  spike_size=20,spike_separator=3)

U.plot_spike_raster([train1,train2],1E-3,1.0;
  spike_size=20,spike_separator=3)

##

const r_pre = 10.0
const r_post = 10.0
const Ttot = 20_000.0

spiketrains = let spikes_shared = U.make_poisson_samples(0.5*r_pre,Ttot)
  spikes_pre = sort(vcat(U.make_poisson_samples(0.5*r_pre,Ttot),spikes_shared))
  spikes_post = sort(vcat(U.make_poisson_samples(0.5*r_post,Ttot),spikes_shared))
  U.SpikeTrains([spikes_pre,spikes_post];t_start=0.0,t_end=Ttot)
end

## check means
therates = U.numerical_rates(spiketrains)

@info """
Presynaptic rate is $(therates[1]) Hz
Postsynaptic rate is $(therates[2]) Hz
"""
##
# check variance too
U.variance_spike_count(spiketrains,1,1.0)
U.variance_spike_count(spiketrains,1,10.0) / 10.0

# okay, now covariance and I am done

U.covariance_spike_count(spiketrains,1,2,1.0)
U.covariance_spike_count(spiketrains,1,2,0.5) / 0.5


U.pearson_correlation(spiketrains,(1,2),2.0)