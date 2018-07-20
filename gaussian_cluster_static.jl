## Define a particle consisting of all cluster variables for a Gaussian mixture
## N is no. of potential clusters
## d is dimension of data
## n_obs is the number of observations
## n_levels is the number of levels, used in multinomial
using StaticArrays
mutable struct sgaussianCluster
  n::MVector
  μ::MMatrix   ## mean of observations in clusters
  Σ::MMatrix   ## sum of observations in clusters
  λ::MMatrix
  β::MMatrix
  c::MVector    ## The allocation vector
  #ζ::Vector{Float64}  ## logprob
  sgaussianCluster(dataFile::Array, N::Int64) =
                                  new(MVector{N, Int64}(zeros(Int64, N)),
                                      MMatrix{N, size(dataFile, 2), Float64}(zeros(Float64, N, size(dataFile, 2))),
                                      MMatrix{N, size(dataFile, 2), Float64}(zeros(Float64, N, size(dataFile, 2))),
                                      MMatrix{N, size(dataFile, 2), Float64}(ones(Float64, N, size(dataFile, 2))),
                                      MMatrix{N, size(dataFile, 2), Float64}(ones(Float64, N, size(dataFile, 2))),
                                      MVector{size(dataFile, 1), Int64}(zeros(Int64, size(dataFile, 1))))
                                      #,Vector{Float64}(zeros(Float64, N)))
end


function calc_logprob!(obs::Array{Float64}, cl::sgaussianCluster, logprob)
  logprob[1] = - 1.310533
   for q in 1:length(obs)
      logprob[1] -=  0.75 * Base.Math.JuliaLibm.log(1.0 + 2.0 * obs[q] ^ 2.0)
  end

  @simd for i = 2:length(logprob)
    logprob[i] = logprob[1]
  end

  # Iterate over clusters
   for i in 1:length(logprob)
      if cl.n[i] > 0
          @inbounds n = cl.n[i] * 0.5 + 0.5
          @fastmath @inbounds logprob[i] = length(obs) * (lgamma(n + 0.5) - lgamma(n) -
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

function particle_add!(obs::Array{Float64}, i::Int64, sstar::Int64, cl::sgaussianCluster)
  @simd for q = 1:length(obs)
      cl.β[sstar, q]  -= 0.5 * (- 2.0 * cl.Σ[sstar, q] * cl.μ[sstar, q] +
                             (cl.μ[sstar, q] ^ 2) * cl.n[sstar]) +
                    (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) * cl.μ[sstar, q] ^ 2
  end
  @inbounds cl.n[sstar]     += Int64(1)
  @inbounds cl.Σ[sstar, :]  += obs
  @simd for q = 1:length(obs)
  @inbounds cl.μ[sstar, q]  = cl.Σ[sstar, q] / (cl.n[sstar] + 1.0)
  cl.β[sstar, q]  += 0.5 * ((obs[q] ^ 2) -
                      2.0 * cl.Σ[sstar, q] * cl.μ[sstar, q] +
                      cl.n[sstar] * cl.μ[sstar, q] ^ 2) +
                      (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) *
                      cl.μ[sstar, q] ^ 2
  @inbounds cl.λ[sstar, q]    = ((0.5 * cl.n[sstar] + 0.5) * (cl.n[sstar] + 1.0)) /
                    (cl.β[sstar, q] * (cl.n[sstar] + 2.0))
  end
  @inbounds cl.c[i]         = Int64(sstar)
  return
end

function particle_reset!(cl::sgaussianCluster)
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

function particle_copy(particle::sgaussianCluster)
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
