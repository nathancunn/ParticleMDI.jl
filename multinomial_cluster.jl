## Define a particle consisting of all cluster variables for a multinomial mixture
## N is no. of potential clusters
## d is dimension of data
## n_obs is the number of observations
## n_levels is the number of levels
mutable struct multinomialCluster{n_obs, d, N, n_levels}
  n::Vector{Int64}
  counts::Array{Int64}   ## count of occurrence of levels
  multinomialCluster{n_obs, d, N, n_levels}() where {n_obs, d, N, n_levels} = new(
                                              Vector{Int64}(zeros(Int64, N)),
                                              Array{Int64}(zeros(Int64, n_levels, d, N)))
end



function calc_logprob!(obs::Array{Float64}, cl::multinomialCluster{n_obs, d, N, n_levels}) where {n_obs, d, N, n_levels}
  # Identify all non-empty clusters
  cl.ζ[1] = - 1.310533
   for q in 1:length(obs)
      cl.ζ[1] -=  0.75 * log(1.0 + 2 * obs[q] ^ 2)
  end
  @simd for n = 1:length(cl.ζ)
    cl.ζ[n] = cl.ζ[1]
  end

  # cl.ζ .= Float64(sum(- 1.310533 .- 0.75 .* log.(1.0 + 2 .* obs .^ 2)))
  # Iterate over clusters
   for i in 1:length(cl.ζ)
      if cl.n[i] > 0
          @inbounds n = cl.n[i] * 0.5 + 0.5
          @fastmath @inbounds cl.ζ[i] = length(obs) * (lgamma(n + 0.5) - lgamma(n) -
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

function particle_add!(obs, i::Int64, sstar::Int64, cl::gaussianCluster{n_obs, d, N, n_levels}) where {n_obs, d, N, n_levels}
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

function particle_reset!(cl::gaussianCluster{n_obs, d, N}) where {n_obs, d, N, n_levels}
  cl.β .= Float64(1)
  cl.λ .= Float64(1)
  cl.μ .= Float64(0)
  cl.Σ .= Float64(0)
  cl.n .= Int64(0)
  # cl.c .= Int64(0)
  # cl.ζ .= Float64(0)
  return
end
