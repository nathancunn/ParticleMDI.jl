## Define a particle consisting of all cluster variables for a Negative Binomial  mixture
## Priors:
# β_0 = 1
# α_0 = 1
# r = 1
mutable struct NegBinomCluster
  n::Int
  Σ::Vector{Int}   ## sum of observations in clusters
  NegBinomCluster(dataFile) =         new(0,
                                      Vector{Int}(zeros(Int, size(dataFile, 2))))
end

function copy_particle(particle::NegBinomCluster, dataFile)
    new_particle = NegBinomCluster(dataFile)
    new_particle.n = particle.n
    for i in eachindex(particle.Σ)
      new_particle.Σ[i] = particle.Σ[i]
    end
    return new_particle
end

function calc_logprob(obs, cl::NegBinomCluster, featureFlag::Array)

    # out = (loggamma(1 + cl.n + 1) - loggamma(1 + cl.n)) * sum(featureFlag)
    out = 0.0
    # Iterate over features
      @inbounds for q in 1:size(obs, 1)
        if featureFlag[q]
        # out += loggamma(1 + obs[q] + cl.Σ[q]) -
        #       loggamma(1 + cl.n + 1 + 1 + obs[q] + cl.Σ[q]) +
        #       loggamma(1 + cl.n + 1 + cl.Σ[q]) -
        #       loggamma(1 + cl.Σ[q])
       out += loggamma(1 + cl.n + 1) +
          loggamma(1 + obs[q] + cl.Σ[q]) +
          loggamma(1 + cl.n + 1 + cl.Σ[q]) -
          loggamma(1 + cl.n + 1 + 1 + obs[q] + cl.Σ[q]) -
          loggamma(1 + cl.n) - loggamma(1 + cl.Σ[q])
        end
      end
      return out
end

function cluster_add!(cl::NegBinomCluster, obs, featureFlag::Array)
  cl.n     += 1
  @inbounds  for q = 1:length(obs)
    if featureFlag[q]
      cl.Σ[q]  += obs[q]
    end
  end
  return
end

function calc_logmarginal(cl::NegBinomCluster)
  # This returns the log of the marginal likelihood of the cluster
  # Used for feature selection
  lm = loggamma.(cl.Σ .+ 1) -
   loggamma.(cl.Σ .+ (cl.n + 1 + 1)) .+
   loggamma(1 + cl.n)
  return lm
end
