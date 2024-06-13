##
push!(LOAD_PATH, abspath(@__DIR__,".."))
using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Makie,CairoMakie
using Random
Random.seed!(0)


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

function draw_spike_raster(S::SpikeTrains{R,N},dt::Real,time_duration::Real;
    t_start::Real=0.0,
    spike_size::Integer = 5,
    spike_separator::Integer = 1,
    background_color::Color=RGB(1.,1.,1.),
    spike_colors::Union{C,Vector{C}}=RGB(0.,0.0,0.0),
    max_size::Real=1E4) where {R,N,C<:Color}


tcorr,corr = U.running_covariance_zero_lag(train1,train2,0.2,1000.)

##

plot(tcorr,corr,linewidth=2.0)


