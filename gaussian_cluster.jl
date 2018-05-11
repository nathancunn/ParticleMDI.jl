## Define a particle consisting of all cluster variables for a Gaussian mixture
## N is no. of potential clusters
## d is dimension of data
## n_obs is the number of observations
mutable struct gaussianCluster{n_obs, d, N}
  n::Vector{Int64}
  μ::Matrix{Float64}   ## mean of observations in clusters
  Σ::Matrix{Float64}   ## sum of observations in clusters
  λ::Matrix{Float64}
  β::Matrix{Float64}
  c::Vector{Int64}    ## The allocation vector
  ζ::Vector{Float64}  ## logprob
  gaussianCluster{n_obs, d, N}() where {n_obs, d, N} = new(
                                              Vector{Int64}(zeros(Int64, N)),
                                              Matrix{Float64}(zeros(Float64, N, d)),
                                              Matrix{Float64}(zeros(Float64, N, d)),
                                              Matrix{Float64}(ones(Float64, N, d)),
                                              Matrix{Float64}(ones(Float64, N, d)),
                                              Vector{Int64}(zeros(Int64, n_obs)),
                                              Vector{Float64}(zeros(Float64, N)))
end


function calc_logprob!(obs::Array{Float64}, cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  # Identify all non-empty clusters
  cl.ζ[1] = - 1.310533
  for q in 1:length(obs)
      cl.ζ[1] -=  0.75 * log(1.0 + 2 * obs[q] ^ 2)
  end
  for n = 1:length(cl.ζ)
    cl.ζ[n] = cl.ζ[1]
  end

  # cl.ζ .= Float64(sum(- 1.310533 .- 0.75 .* log.(1.0 + 2 .* obs .^ 2)))
  # Iterate over clusters
  for i in 1:length(cl.ζ)
      if cl.n[i] > 0
          @inbounds n = cl.n[i] * 0.5 + 0.5
          cl.ζ[i] = length(obs) * (lgamma(n + 0.5) - lgamma(n) -
                    0.5723649429247001)
        # Iterate over features
          for q in 1:length(obs)
              cl.ζ[i] += 0.5 * Base.Math.JuliaLibm.log(cl.λ[i, q] / n) -
                        (n + 0.5) * Base.Math.JuliaLibm.log((1.0 / n) *
                        (obs[q] - cl.μ[i, q]) ^ 2.0 * cl.λ[i, q] + 1.0)

          end
      end
  end
    return
end

function particle_add(obs, i::Int64, sstar::Int64, cl_old::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
    cl = deepcopy(cl_old)
    for q = 1:length(obs)
        cl.β[sstar, q]  -= 0.5 * (- 2.0 * cl.Σ[sstar, q] * cl.μ[sstar, q] +
                               (cl.μ[sstar, q] ^ 2) * cl.n[sstar]) +
                      (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) * cl.μ[sstar, q] ^ 2
    end
    @inbounds cl.n[sstar]     += Int64(1)
    @inbounds cl.Σ[sstar, :]  += obs
    for q = 1:length(obs)
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
  return cl
end

function particle_reset!(cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  cl.β .= Float64(1)
  cl.λ .= Float64(1)
  cl.μ .= Float64(0)
  cl.Σ .= Float64(0)
  cl.n .= Int64(0)
  cl.c .= Int64(0)
  cl.ζ .= Float64(0)
  return
end


function particle_remove(obs, sstar::Int64, p::Int64, cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  cl.c            = setindex(cl.c, 0, p)
  cl.β[sstar, :]  -= 0.5 * ((obs .^ 2) - 2 * cl.Σ[sstar, :] .* cl.μ[sstar, :] .+
                     cl.n[sstar] .* cl.μ[sstar, :] .^ 2) .-
                     (cl.n[sstar] / (2 * (cl.n[sstar] + 1))) .* cl.μ[sstar, :] .^ 2
  cl.n            = setindex(cl.n, cl.n[sstar] - 1, sstar)

  if cl.n[sstar] == 0
    cl.Σ[sstar, :] = 0
    cl.μ[sstar, :] = 0
    cl.β[sstar, :] = 1
    cl.λ[sstar, :] = 1
  elseif cl.n[sstar] < 0
    error("Found cluster with negative values")
  else
    cl.Σ[sstar, :]  -= obs
    cl.μ[sstar, :]  = cl.Σ[sstar, :] ./ (cl.n[sstar] + 1)
    cl.β[sstar, :]  += 0.5 .* (- 2 * cl.Σ[sstar, :] .* cl.μ[sstar, :] +
                              cl.n[sstar] .* cl.μ[sstar, :] .^ 2) .+
                      (cl.n[sstar] / (2 * (cl.n[sstar] + 1))) .* cl.μ[sstar, :] .^ 2
    cl.λ[sstar, :]    = ((0.5 * cl.n[sstar] + 0.5) .* (cl.n[sstar] + 1)) ./
                      (cl.β[sstar, :] .* (cl.n[sstar] + 2))

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
