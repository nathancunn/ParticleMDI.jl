## Define a particle consisting of all cluster variables for a Gaussian mixture
## N is no. of potential clusters
## d is dimension of data
## n_obs is the number of observations
## n_levels is the number of levels, used in multinomial
mutable struct GaussianCluster
  n::Int64
  μ::Vector{Float64}   ## mean of observations in clusters
  Σ::Vector{Float64}   ## sum of observations in clusters
  λ::Vector{Float64}
  β::Vector{Float64}
  GaussianCluster(dataFile) =
                                  new(0,
                                      Vector{Float64}(zeros(Float64, size(dataFile, 2))),
                                      Vector{Float64}(zeros(Float64, size(dataFile, 2))),
                                      Vector{Float64}(ones(Float64, size(dataFile, 2))),
                                      Vector{Float64}(ones(Float64, size(dataFile, 2))))
end

function calc_logprob(obs::Array, cl::GaussianCluster)
  # Iterate over clusters
  n = cl.n * 0.5 + 0.5
  out = length(obs) * (Base.Math.JuliaLibm.lgamma((n) + 0.5) -
                                                Base.Math.JuliaLibm.lgamma(n) -
                                                0.5723649429247001)
  # Iterate over features
    for q in 1:length(obs)
        out +=  0.5 * Base.Math.JuliaLibm.log(cl.λ[q] / n) -
                            (n + 0.5) * Base.Math.JuliaLibm.log((1.0 / n) *
                            (obs[q] - cl.μ[q]) ^ 2.0 * cl.λ[q] + 1.0)
    end

    return out
end

function cluster_add!(cl::GaussianCluster, obs::Array)
  @inbounds for q = 1:length(obs)
    cl.β[q]  -= 0.5 * (- 2.0 * cl.Σ[q] * cl.μ[q] +
                             (cl.μ[q] ^ 2) * cl.n) +
                    (cl.n / (2.0 * (cl.n + 1.0))) * cl.μ[q] ^ 2
  end
  @inbounds cl.n     += 1
  @inbounds for q = 1:length(obs)
    cl.Σ[q]  += obs[q]
    cl.μ[q]  = cl.Σ[q] / (cl.n + 1.0)
    cl.β[q]  += 0.5 * ((obs[q] ^ 2) -
                      2.0 * cl.Σ[q] * cl.μ[q] +
                      cl.n * cl.μ[q] ^ 2) +
                      (cl.n / (2.0 * (cl.n + 1.0))) *
                      cl.μ[q] ^ 2
    cl.λ[q]    = ((0.5 * cl.n + 0.5) * (cl.n + 1.0)) /
                    (cl.β[q] * (cl.n + 2.0))
  end
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
