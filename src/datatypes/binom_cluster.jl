## Define a particle consisting of all cluster variables for a Binomial mixture
## Assumes N is 25
mutable struct BinomCluster
  n::Int                          # count of cluster members
  α::Array{Int}     # count of occurrence of levels
  β::Array{Int}           # The denominator in logprob calculation
  BinomCluster(dataFile::Matrix{Int}) = new(0,
                                              ones(Int, size(dataFile, 2)),
                                              ones(Int, size(dataFile, 2)))
end


function copy_particle(particle::BinomCluster, dataFile)
    new_particle = BinomCluster(dataFile)
    new_particle.n = particle.n
    for i in eachindex(particle.α)
      new_particle.α[i] = particle.α[i]
      new_particle.β[i] = particle.β[i]
    end
    return new_particle
end

function calc_logprob(obs::Array{Int}, cl::BinomCluster)
  out = 0.0
  for q in 1:length(obs)
    @inbounds out += - loggamma(obs[q] + 1) - loggamma(25 - obs[q] + 1) +
                    loggamma(cl.α[q] + obs[q]) + loggamma(25 - obs[q] + cl.β[q]) +
                    loggamma(cl.α[q] + cl.β[q]) -
                    loggamma(cl.α[q] + 25 + cl.β[q]) - loggamma(cl.α[q]) - loggamma(cl.β[q])
  end
  return out
end

function cluster_add!(cl::BinomCluster, obs::Array{Int})
  @inbounds cl.n  += Int(1)
  @inbounds cl.α  += obs
  @inbounds cl.β  += 25 .- obs
end
