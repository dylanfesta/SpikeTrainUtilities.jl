using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Test
using Statistics, Distributions
using Random ; Random.seed!(0)
import StatsBase: midpoints



@testset "Concatenate trains" begin

  n = 10
  rates1 = 10 .* rand(n) 
  rates2 = 20 .* rand(n)
  rates3 = 5 .* rand(n)

  Δt1 = 30.0
  Δt2 = 60.0
  Δt3 = 10.0

  train1 = U.make_random_spiketrains(rates1,Δt1)
  train2 = U.make_random_spiketrains(rates2,Δt2;t_start=Δt1)
  train3 = U.make_random_spiketrains(rates3,Δt3;t_start=Δt1+Δt2)

  @test maximum(train1) < Δt1
  @test minimum(train2) > Δt1

  trains_all = U.cat(train1,train2,train3)

  @test trains_all.t_start == 0.0
  @test trains_all.t_end == Δt1+Δt2+Δt3
  @test minimum(trains_all) == minimum(train1)
  @test maximum(trains_all) == maximum(train3)
end

#= @testset "Merge trains" begin
  n1,n2,n3 = 10,3,22
  rates1 = 10 .* rand(n1) 
  rates2 = 20 .* rand(n2)
  rates3 = 5 .* rand(n3)

  Ttot = 50.0

  train1 = U.make_random_spiketrains(rates1,Ttot)
  train2 = U.make_random_spiketrains(rates2,Ttot)
  train3 = U.make_random_spiketrains(rates3,Ttot)


  train1merged = U.merge(train1)

  @test train1merged.n_units == n1
  @test all(train1merged.trains[end] == train1merged.trains[end])

  trainsallmerged = U.merge(train1,train2,train3)
  @test trainsallmerged.n_units == n1+n2+n3
  @test all(trainsallmerged.trains[n1+1] .== train2.trains[1])
  @test all(trainsallmerged.trains[end] .== train3.trains[end])
end =#

@testset "Numerical rates" begin

  n = 10
  rates1 = 60 .* rand(n) 
  T = 500.0

  trains1 = U.make_random_spiketrains(rates1,T)
  rates_num = U.numerical_rates(trains1)
  @test all(isapprox.(rates1,rates_num,rtol=0.1))

  T = 1000.0
  trains2 = U.make_random_spiketrains(rates1,T)
  rates_num2 = U.numerical_rates(trains2;t_start=500.0)
  @test all(isapprox.(rates1,rates_num,rtol=0.1))

end

@testset "Spike binning with arbitrary borders" begin

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
  @test all(n_spikes1_check .== n_spikes1)

  _, n_spikes_check = U.bin_spikes(_spiketrain,borders)
  @test all(n_spikes_check[:,1] .== n_spikes1)
  @test all(n_spikes_check[:,2] .== n_spikes2)

  mid_p, instrates = U.instantaneous_rates(_spiketrain,borders)

  @test all(isapprox.(mid_p,midpoints(borders)))
  @test all(isapprox.(instrates[:,1],n_spikes1./diff(borders)))
  @test all(isapprox.(instrates[:,2],n_spikes2./diff(borders)))

end

@testset "Second order" begin
  n = 2
  rates1 = 60 .* rand(n) 
  T = 1000.0
  trains1 = U.make_random_spiketrains(rates1,T)
  covt,covij = U.covariance_density_ij(trains1,(1,2),0.03,2.0)
  traintest1 = trains1.trains[1]
  mystd = 0.2
  traintest2 = filter!(>(0.0),traintest1 .+ mystd.*randn(length(traintest1)))
  covt,covij = U.covariance_density_ij(traintest1,traintest2,0.02,1.0)
  @test sum(covij) > 100.0



  rate = 100.0
  T = 5_000.0
  train1 = U.make_poisson_samples(rate,T)

  train2 = let ret = Float64[]
    t_now = 0.0
    t_end = train1[end]
    for tt in train1
      if rand() > (tt/t_end -0.5)
        Δt = -log(rand())/rate
        t_now += Δt
      else
        t_now = tt + randn()*0.5
      end
      push!(ret,t_now)
    end
    sort(ret)
  end

  tcorr,corr = U.running_covariance_zero_lag(train1,train2,0.5,500.)
  nc = length(corr)
  @test abs(mean(corr[1:nc÷2])) <  10*abs(mean(corr[nc÷2+1:end]))

end

@testset "Fano factor" begin
  Ttot= 10_000.0
  therates=[13.,107.,523.]
  trains = U.make_random_spiketrains(therates,Ttot)

  # variances should be close to means, for different time bins!
  Δt_test = [1.0,10.0,20.0]

  for neu_test in (1,2,3)
    _expected_vars = therates[neu_test].*Δt_test 
    _num_vars = [ U.variance_spike_count(trains,neu_test,Δt) for Δt in Δt_test]
    for (ev,nv) in zip(_expected_vars,_num_vars)
      @test isapprox(ev,nv,rtol=0.1)
    end
    _fanos = [ U.fano_factor_spike_count(trains,neu_test,Δt) for Δt in Δt_test]
    for _f in _fanos
      @test isapprox(_f,1.0,rtol=0.1)
    end
    _cv = U.coefficient_of_variation(trains,neu_test)
    @test isapprox(_cv,1.0,rtol=0.1)
  end
end


@testset "Counting spikes, using structs" begin
  n  = 44
  rates1 = 60 .* rand(n) 
  T = 1000.0
  dt = 0.1
  trains1 = U.make_random_spiketrains(rates1,T)
  trains_bincounts = U.discretize(trains1,dt,U.BinCount())
  edges_test1 = U.get_t_edges(trains1,dt)
  edges_test2 = U.get_t_edges(trains_bincounts)
  @test all(edges_test1 .== edges_test2)
  for (k,_train) in enumerate(trains1.trains)
    _train_lesstime = _train[_train .< trains_bincounts.t_end]
    @test length(_train_lesstime) == sum(trains_bincounts.ys[k,:])
  end
end

@testset "Instantaneous firing rates, and shuffling" begin
  
  n  = 44
  ratestest = 250 .* rand(n) 
  T = 1000.0
  trainstest = U.make_random_spiketrains(ratestest,T)

  # compute iFR with cutoff at 5ms
  iFRs = U.get_instantaneous_firing_rates(trainstest; dt_min=5E-3)

  # test they are all below 200 Hz
  for y in iFRs.ys
    @test all(<=(200.0),y)
  end

  trains_shifted = U.circular_shift_of_spiketrains(trainstest)[1]

  for neu in 1:n
    t1 = trainstest.trains[neu]
    t2 = trains_shifted.trains[neu]
    @test issorted(t1)
    @test issorted(t2)
    @test all(t1 .!= t2) 
  end

end

@testset "Sorting units in place" begin

  my_units = collect('a':'z')
  n = length(my_units)

  the_rates = 60 .* rand(n) 
  T = 100.0
  trains = [U.make_poisson_samples(rate,T) for rate in the_rates ]

  spiketrains = U.SpikeTrains(trains; t_start=0.0, t_end=T,
    units=my_units)


  units_shuffled1 = shuffle(my_units)
  resorting_dict1 = Dict(zip(units_shuffled1,1:n))
  units_shuffled2 = shuffle(my_units)
  resorting_dict2 = Dict(zip(units_shuffled2,0:n-1))

  U.sort_units!(spiketrains,resorting_dict1)

  @test all(spiketrains.units .== units_shuffled1)
  U.sort_units!(spiketrains,resorting_dict2)
  @test all(spiketrains.units .== units_shuffled2)
end
