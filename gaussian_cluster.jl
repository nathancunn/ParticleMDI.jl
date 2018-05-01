## Define a particle consisting of all cluster variables for a Gaussian mixture
## N is no. of potential clusters
## d is dimension of data
## n_obs is the number of observations
mutable struct gaussianCluster{n_obs, d, N}
  # n::SVector{N, Int64}        ## number of obs in a cluster
  # μ::MMatrix{N, d, Float64}   ## mean of observations in clusters
  # Σ::MMatrix{N, d, Float64}   ## sum of observations in clusters
  # λ::MMatrix{N, d,  Float64}
  # β::MMatrix{N, d, Float64}
  # c::SVector{n_obs, Int64}    ## The allocation vector
  n::Vector{Int64}
  μ::Matrix{Float64}   ## mean of observations in clusters
  Σ::Matrix{Float64}   ## sum of observations in clusters
  λ::Matrix{Float64}
  β::Matrix{Float64}
  c::Vector{Int64}    ## The allocation vector
  ζ::Vector{Float64}  ## logprob
  gaussianCluster{n_obs, d, N}() where {n_obs, d, N} = new(
                                              # Int64(1),
                                              # SVector{N, Int64}(zeros(Int64, N)),
                                              # MMatrix{N, d, Float64}(zeros(Float64, N, d)),
                                              # MMatrix{N, d, Float64}(zeros(Float64, N, d)),
                                              # MMatrix{N, d, Float64}(ones(Float64, N, d)),
                                              # MMatrix{N, d, Float64}(ones(Float64, N, d)),
                                              # SVector{n_obs, Int64}(zeros(Int64, n_obs)),
                                              Vector{Int64}(zeros(Int64, N)),
                                              Matrix{Float64}(zeros(Float64, N, d)),
                                              Matrix{Float64}(zeros(Float64, N, d)),
                                              Matrix{Float64}(ones(Float64, N, d)),
                                              Matrix{Float64}(ones(Float64, N, d)),
                                              Vector{Int64}(zeros(Int64, n_obs)),
                                              Vector{Float64}(zeros(Float64, N)))
end


function calc_logprob!(obs::Vector{Float64}, cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  # cl.ζ .= sum(lgamma(0.75) - lgamma(0.25) + 0.5 * log(1 / (0.5 * π)) -
  #                    0.75 * log.(1 + 2 * (obs) .^ 2), 1)[1]
  cl.ζ .= sum(-1.310533 - 0.75 * log.(1.0 + 2 * obs .^ 2))
  # Identify all non-empty clusters
  ind = Vector{Bool}(cl.n .!= 0)
  n = cl.n[ind] .* 0.5 .+ 0.5
  λ = cl.λ[ind, :]

  # Posterior predictive taking normal-gamma prior
  cl.ζ[ind] = sum(lgamma.(n + 0.5) .-
                      lgamma.(n) .+
                      0.5 .* log.(λ ./ n) .-
                      (n .+ 0.5) .* log.((1.0 ./ n) .*
                      (- cl.μ[ind, :] .+ transpose(obs)) .^ 2.0 .* λ .+ 1.0) .-
                      0.5723649429247001 ## 0.5 * log(π)
                  , 2)
    return
    return
end


function particle_add(obs, i, sstar::Int64, cl_old::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  cl = deepcopy(cl_old)
    cl.β[sstar, :]  -= 0.5 .* (- 2.0 * cl.Σ[sstar, :] .* cl.μ[sstar, :] +
                              cl.n[sstar] .* cl.μ[sstar, :] .^ 2) .+
                      (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) .* cl.μ[sstar, :] .^ 2
    cl.n[sstar]     += Int64(1)
    cl.Σ[sstar, :]  += obs
    cl.μ[sstar, :]  = cl.Σ[sstar, :] ./ (cl.n[sstar] + 1.0)
    cl.β[sstar, :]  += 0.5 * ((obs .^ 2) - 2.0 * cl.Σ[sstar, :] .* cl.μ[sstar, :] .+
                        cl.n[sstar] .* cl.μ[sstar, :] .^ 2) .+
                        (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) .* cl.μ[sstar, :] .^ 2
    cl.λ[sstar, :]    = ((0.5 * cl.n[sstar] + 0.5) .* (cl.n[sstar] + 1.0)) ./
                      (cl.β[sstar, :] .* (cl.n[sstar] + 2.0))
    cl.c[i]         = Int64(sstar)
  return cl
end

function particle_reset!(cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  cl.β .= Float64(1)
  cl.λ .= Float64(1)
  cl.μ .= Float64(0)
  cl.Σ .= Float64(0)
  cl.n .= Int64(0)
  cl.c .= Int64(0)
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
  μ = mapslices(median, dataFile, 1)
  σ = 0.5 .* μ - mapslices(x -> quantile(x, 0.05), dataFile, 1) + eps(Float64)
  dataFile = (dataFile .- μ) ./ σ
end
