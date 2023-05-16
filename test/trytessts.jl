##
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Test
using Plots

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

tcorr,corr = U.running_covariance_zero_lag(train1,train2,0.2,1000.)

##

plot(tcorr,corr,linewidth=2.0)


