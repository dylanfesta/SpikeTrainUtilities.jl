using SpikeTrainUtilities ; global const U = SpikeTrainUtilities
using Test
using Statistics
using Random ; Random.seed!(0)

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