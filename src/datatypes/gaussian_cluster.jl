## Define a particle consisting of all cluster variables for a Gaussian mixture
## N is no. of potential clusters
## d is dimension of data
## n_obs is the number of observations
## n_levels is the number of levels, used in multinomial
mutable struct GaussianCluster
  n::Vector{Int64}
  μ::Matrix{Float64}   ## mean of observations in clusters
  Σ::Matrix{Float64}   ## sum of observations in clusters
  λ::Matrix{Float64}
  β::Matrix{Float64}
  c::Vector{Int64}    ## The allocation vector
  #ζ::Vector{Float64}  ## logprob
  GaussianCluster(dataFile::Array, N::Int64) =  new(Vector{Int64}(zeros(Int64, N)),
                                      Matrix{Float64}(zeros(Float64, N, size(dataFile, 2))),
                                      Matrix{Float64}(zeros(Float64, N, size(dataFile, 2))),
                                      Matrix{Float64}(ones(Float64, N, size(dataFile, 2))),
                                      Matrix{Float64}(ones(Float64, N, size(dataFile, 2))),
                                      Vector{Int64}(zeros(Int64, size(dataFile, 1))))
                                      #,Vector{Float64}(zeros(Float64, N)))
end


function calc_logprob!(logprob::SubArray, obs::Array, cl::particleMDI.GaussianCluster)
  logprob[1] = - 1.310533
   for obs_q in obs
      logprob[1] -=  0.75 * Base.Math.JuliaLibm.log(1.0 + 2.0 * obs_q ^ 2.0)
  end

  @simd for i = 2:length(logprob)
    @inbounds logprob[i] = logprob[1]
  end

  # Iterate over clusters
  Q = length(obs)
   for i in 1:length(logprob)
      if cl.n[i] > 0
          @inbounds n = cl.n[i] * 0.5 + 0.5
          @fastmath @inbounds logprob[i] = Q * (lgamma(n + 0.5) - lgamma(n) -
                              0.5723649429247001)
        # Iterate over features
         @inbounds for q in 1:length(obs)
              logprob[i] += 0.5 * Base.Math.JuliaLibm.log(cl.λ[i, q] / n) -
                                  (n + 0.5) * Base.Math.JuliaLibm.log((1.0 / n) *
                                  (obs[q] - cl.μ[i, q]) ^ 2.0 * cl.λ[i, q] + 1.0)

          end
      end
  end
    return
end

function particle_add!(cl::particleMDI.GaussianCluster, obs::Array, i::Int64, sstar::Int64)
  @inbounds @simd for q = 1:length(obs)
    cl.β[sstar, q]  -= 0.5 * (- 2.0 * cl.Σ[sstar, q] * cl.μ[sstar, q] +
                             (cl.μ[sstar, q] ^ 2) * cl.n[sstar]) +
                    (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) * cl.μ[sstar, q] ^ 2
  end
  @inbounds cl.n[sstar]     += Int64(1)
  @inbounds cl.Σ[sstar, :]  += obs
  @inbounds @simd for q = 1:length(obs)
    cl.μ[sstar, q]  = cl.Σ[sstar, q] / (cl.n[sstar] + 1.0)
    cl.β[sstar, q]  += 0.5 * ((obs[q] ^ 2) -
                      2.0 * cl.Σ[sstar, q] * cl.μ[sstar, q] +
                      cl.n[sstar] * cl.μ[sstar, q] ^ 2) +
                      (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) *
                      cl.μ[sstar, q] ^ 2
    cl.λ[sstar, q]    = ((0.5 * cl.n[sstar] + 0.5) * (cl.n[sstar] + 1.0)) /
                    (cl.β[sstar, q] * (cl.n[sstar] + 2.0))
  end
  @inbounds cl.c[i]         = sstar
  return
end

function particle_reset!(cl::particleMDI.GaussianCluster)
  cl.β .= Float64(1)
  cl.λ .= Float64(1)
  cl.μ .= Float64(0)
  cl.Σ .= Float64(0)
  cl.n .= Int64(0)
  # cl.c .= Int64(0)
  # cl.ζ .= Float64(0)
  return
end


function gaussian_normalise!(dataFile::Array)
  for d = 1:size(dataFile, 2)
    # μ = mean(dataFile[:, d])
    # σ = std(dataFile[:, d]) + eps(Float64)
    μ = median(dataFile[:, d])
    σ = 0.5 * μ - quantile(dataFile[:, d], 0.05) + eps(Float64)
    dataFile[:, d] = (dataFile[:, d] .- μ) ./ σ
  end
  return
end

function particle_copy(particle::particleMDI.GaussianCluster)
    # As the mutation step involves copy the contents of a particle
    # This is slightly more efficient than deepcopy
    out = typeof(particle)()
    out.μ = particle.μ
    out.n = particle.n
    out.Σ = particle.Σ
    out.λ = particle.λ
    out.β = particle.β
    out.c = particle.c
    # out.ζ = particle.ζ
    return out
end
