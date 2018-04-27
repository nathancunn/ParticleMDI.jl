## Define a particle consisting of all cluster variables for a Gaussian mixture
## N is no. of potential clusters
## d is dimension of data
## n_obs is the number of observations
mutable struct gaussianCluster{n_obs, d, N}
  # n is no. of clusters, d is dimension of data
  # c::Int64 ## The cluster index
  # n::SVector{N, Int64}        ## number of obs in a cluster
  n::Vector{Int64}
  # μ::MMatrix{N, d, Float64}   ## mean of observations in clusters
  # Σ::MMatrix{N, d, Float64}   ## sum of observations in clusters
  # λ::MMatrix{N, d,  Float64}
  # β::MMatrix{N, d, Float64}
  μ::Matrix{Float64}   ## mean of observations in clusters
  Σ::Matrix{Float64}   ## sum of observations in clusters
  λ::Matrix{Float64}
  β::Matrix{Float64}
  # c::SVector{n_obs, Int64}    ## The allocation vector
  c::Vector{Int64}    ## The allocation vector
  ζ::Vector{Float64}        ## logprob
  gaussianCluster{n_obs, d, N}() where {n_obs, d, N} = new(
                                              # Int64(1),
                                              # SVector{N, Int64}(zeros(Int64, N)),
                                              Vector{Int64}(zeros(Int64, N)),
                                              # MMatrix{N, d, Float64}(zeros(Float64, N, d)),
                                              # MMatrix{N, d, Float64}(zeros(Float64, N, d)),
                                              # MMatrix{N, d, Float64}(ones(Float64, N, d)),
                                              # MMatrix{N, d, Float64}(ones(Float64, N, d)),
                                              Matrix{Float64}(zeros(Float64, N, d)),
                                              Matrix{Float64}(zeros(Float64, N, d)),
                                              Matrix{Float64}(ones(Float64, N, d)),
                                              Matrix{Float64}(ones(Float64, N, d)),
                                              # SVector{n_obs, Int64}(zeros(Int64, n_obs)),
                                              Vector{Int64}(zeros(Int64, n_obs)),
                                              Vector{Float64}(zeros(Float64, N)))
end


function calc_logprob!(obs::Vector{Float64}, cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  # cl.ζ .= sum(lgamma(0.75) - lgamma(0.25) + 0.5 * log(1 / (0.5 * π)) -
  #                    0.75 * log.(1 + 2 * (obs) .^ 2), 1)[1]
  cl.ζ .= sum(-1.310533 - 0.75 * log.(1.0 + 2 * obs .^ 2))
  # Identify all non-empty clusters
  ind = SVector{N, Bool}(cl.n .!= 0)
  cl.ζ[ind] = sum(lgamma.((cl.n[ind] + 2.0) / 2.0) .-
                      lgamma.((1.0 + cl.n[ind]) / 2.0) .+
                      0.5 .* log.(cl.λ[ind, :] ./ ((0.5 + 0.5 .* cl.n[ind]) .* π)) -
                      ((cl.n[ind] + 2.0) ./ 2.0) .* log.(1.0 + (1.0 ./ (0.5 + cl.n[ind] .* 0.5)) .*
                      (-cl.μ[ind, :] .+ transpose(obs)) .^ 2.0 .* cl.λ[ind, :])
                  , 2)
    return
end


function particle_add!(obs, i, sstar::Int64, cl1::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  # cl.c            = sstar
  cl = deepcopy(cl1)
  cl.β[sstar, :]  -= 0.5 .* (- 2.0 * cl.Σ[sstar, :] .* cl.μ[sstar, :] +
                            cl.n[sstar] .* cl.μ[sstar, :] .^ 2) .+
                    (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) .* cl.μ[sstar, :] .^ 2
  # cl.n            = setindex(cl.n, cl.n[sstar] + 1, sstar)
  cl.n[sstar]     += Int64(1)
  cl.Σ[sstar, :]  += obs
  cl.μ[sstar, :]  = cl.Σ[sstar, :] ./ (cl.n[sstar] + 1.0)
  cl.β[sstar, :]  += 0.5 * ((obs .^ 2) - 2.0 * cl.Σ[sstar, :] .* cl.μ[sstar, :] .+
                      cl.n[sstar] .* cl.μ[sstar, :] .^ 2) .+
                      (cl.n[sstar] / (2.0 * (cl.n[sstar] + 1.0))) .* cl.μ[sstar, :] .^ 2
  cl.λ[sstar, :]    = ((0.5 * cl.n[sstar] + 0.5) .* (cl.n[sstar] + 1.0)) ./
                    (cl.β[sstar, :] .* (cl.n[sstar] + 2.0))
  # cl.c            = setindex(cl.c, sstar, i)
  cl.c[i]         = Int64(sstar)
  return cl
end

function particle_reset!(cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N}
  # fill!(cl.β, 1)
  # fill!(cl.λ, 1)
  # fill!(cl.μ, 0)
  # fill!(cl.Σ, 0)
  cl.β .= Float64(1)
  cl.λ .= Float64(1)
  cl.μ .= Float64(0)
  cl.Σ .= Float64(0)
  cl.n .= Int64(0)
  cl.c .= Int64(0)
  # cl.n = SVector{N, Int64}(zeros(Int64, N))
  # cl.c = SVector{n_obs, Int64}(zeros(Int64, n_obs))
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
