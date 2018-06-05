## Define a particle consisting of all cluster variables for a multinomial mixture
mutable struct multinomialCluster
  n::Vector{Int64}        # count of cluster members
  counts::Array{Int64}    # count of occurrence of levels
  c::Vector{Int64}        # The allocation vector
  ζ::Vector{Float64}      # logprob
  multinomialCluster(dataFile, N) = new(Vector{Int64}(zeros(Int64, N)),
                                        Array{Int64}(zeros(Int64, maximum(dataFile),
                                                           size(dataFile, 2), N)),
                                        Vector{Int64}(zeros(Int64, size(dataFile, 1))),
                                        Vector{Float64}(zeros(Float64, N)))
end



function calc_logprob!(obs::Array{Int64}, cl::multinomialCluster)
  # Identify all non-empty clusters
  cl.ζ[1] = log(0.5) * length(obs) - log(0.5 * size(cl.counts, 3)) * length(obs)
  @simd for n = 2:length(cl.ζ)
    cl.ζ[n] = cl.ζ[1]
  end

  # Iterate over clusters
   for i in 1:length(cl.ζ)
      if cl.n[i] > 0
        cl.ζ[i] = 0
        for q in 1:length(obs)
          cl.ζ[i] += log(0.5 + cl.counts[obs[q], q, i]) - log(0.5 * size(cl.counts, 3) + sum(cl.counts[:, q, i]))
          end
      end
  end
    return
end

function particle_add!(obs::Array{Int64}, i::Int64, sstar::Int64, cl::multinomialCluster)
  @inbounds cl.n[sstar]     += Int64(1)
  @simd for q = 1:length(obs)
    cl.counts[obs[q], q, sstar] += Int64(1)
  end
    @inbounds cl.c[i]         = Int64(sstar)
  return
end

function particle_reset!(cl::multinomialCluster)
  cl.counts .= Int64(0)
  cl.n .= Int64(0)
  return
end
