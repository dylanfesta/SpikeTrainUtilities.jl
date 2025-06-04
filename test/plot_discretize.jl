
push!(LOAD_PATH, abspath(@__DIR__,".."))
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Makie,CairoMakie
using Random
using Distributions
Random.seed!(0);
import StatsBase: midpoints


##
n = 3
rates = 10 .* rand(n)

spiketrains = U.make_random_spiketrains(rates,5.0)
sigma_gauss = 0.2
dt_discretize = 0.1

spiketrains_discrete = U.discretize(spiketrains,dt_discretize,U.BinCausalGaussianKernel(sigma_gauss))

##

# plot the thing
function dotheplot()
  fig = Figure(size=(800,600))
  ax = Axis(fig[1,1], title="Discretized Spike Trains", xlabel="Time (s)", ylabel="Spike Count")
  times = U.get_t_midpoints(spiketrains_discrete)
  kplot = 2
  ys = spiketrains_discrete.ys[kplot,:]
  lines!(ax, times, ys, color=:blue, label="Discretized Spikes")
  spiketimes = spiketrains.trains[kplot]
  nspikes = length(spiketimes)
  if nspikes > 0
    scatter!(ax, spiketimes, fill(0.5, nspikes), color=:black, label="Original Spikes", markersize=10)
  end
  fig
end

thefig = dotheplot()