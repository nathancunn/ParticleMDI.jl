using particleMDI
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here
@test 1 == 1

## Test data types
# Gaussian
using Distributions
test_data = rand(Normal(0, 10), 1000, 1)
test_cluster = particleMDI.GaussianCluster(test_data)
# Basic check it's been initialised
@test test_cluster.n == 0
# Check the logprob
# particleMDI.calc_logprob(test_data[1, :], test_cluster)
for obs in test_data
    particleMDI.cluster_add!(test_cluster, [obs])
end

@test test_cluster.n == 1000
@test isapprox(test_cluster.Σ[1], sum(test_data))
@test isapprox(test_cluster.μ[1], test_cluster.Σ[1] / (1001))
xbar = test_cluster.Σ[1] / 1001
s2 = sum((test_data .- xbar) .^ 2)
beta = 1 + 0.5 * (s2 + ((1. + size(test_data, 1)) ^ - 1) * (1 * size(test_data, 1)) * (- xbar) ^ 2)
@test isapprox(test_cluster.β[1], beta)
@test isapprox(test_cluster.λ[1], (500.5 * 1001) / (test_cluster.β[1] * 1002))

xcentred = (test_data[end, :][1] - test_cluster.μ[1]) * sqrt(test_cluster.λ[1])
truelogprob = logpdf(TDist(test_cluster.n + 1),xcentred) + 0.5 * log(test_cluster.λ[1])
estlogprob = particleMDI.calc_logprob(test_data[end, :], test_cluster)
@test isapprox(truelogprob, estlogprob)

# Categorical
test_data = rand(1:10, 1000, 1)
test_cluster = particleMDI.CategoricalCluster(test_data)
@test test_cluster.n == 0

for obs in test_data
    particleMDI.cluster_add!(test_cluster, [obs])
end

@test test_cluster.n == 1000
test_cluster.counts
for x in unique(test_data)
    @test sum(test_data .== x) == test_cluster.counts[1][x]
end

@test particleMDI.calc_logprob([1], test_cluster) ==
      log((sum(test_data .== 1) + 0.5) / (1005))

      
