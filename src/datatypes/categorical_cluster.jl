## Define a particle consisting of all cluster variables for a Categorical mixture
mutable struct CategoricalCluster
  n::Int64                          # count of cluster members
  counts::Vector{Vector{Int64}}     # count of occurrence of levels
  nlevels::Array{Float64}           # The denominator in logprob calculation
  CategoricalCluster(dataFile::Matrix{Int64}) = new(0,
                                        [zeros(Int64, maximum(dataFile[:, d])) for d = 1:size(dataFile, 2)],
                                        0.5 * mapslices(maximum, dataFile, dims = 1))
end

function calc_logprob(obs::Array{Int64}, cl::CategoricalCluster)
  out = 0.0
  for nlev in cl.nlevels
    @fastmath out -= log(nlev + cl.n)
  end
#  out = - sum(log.(cl.nlevels .+ cl.n))
  for q in 1:length(obs)
    if cl.n == 0
      @inbounds out += log(0.5)
    else
      @inbounds out += log(0.5 + cl.counts[q][obs[q]])
    end
  end
  return out
end

function cluster_add!(cl::CategoricalCluster, obs::Array{Int64})
  @inbounds cl.n  += Int64(1)
  @simd for q = 1:length(obs)
    @inbounds cl.counts[q][obs[q]] += Int64(1)
  end
  return
end
