# Implement particle cluster analysis for univariate Gaussian data
## Load the packages
using SequentialMonteCarlo
using StaticArrays

include("particles.jl")

function pMDImodel(data, dataTypes, N)
    K       = length(data) # No. of datasets
    n_obs   = size(data[1])[1]
    @assert length(dataTypes) == K
    @assert all(x->x==n_obs, [size(data[k])[1] for k = 1:K])

    @inline function M!(newParticle::mdiParticle{K}, rng::SMCRNG, p::Int64,
      particle::mdiParticle{K}, scratch::Void) where {K}
      if p == 1
        for k = 1:K
            # Set the particle to a container for the corresponding data type
            newParticle.clusters[k] = dataTypes[k]{n_obs, d[k], N}()
            # Add the first observation to cluster 1
            particle_add(data[k][1, :], 1, 1, newParticle.clusters[k])
        end
      else
          #newParticle = particle
          if smcio.resample[p - 1] == true
              # Reset particle weights
              newParticle.logW = 0
              # Reallocate previous observations
              for j = 1:(p - 1)
                  sstar             = [newParticle.clusters[k].c[j] for k = 1:K]
                  #print(newParticle.clusters[1].c[1:p], "\n")
                  for k = 1:K
                      particle_remove(data[k][j, :], sstar[k], j, newParticle.clusters[k])
                  end
                  #print([newParticle.clusters[k].n for k = 1:K], "\n")
                  logprob           = [calc_logprob(data[k][j, :], newParticle.clusters[k]) for k = 1:K]
                  fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
                  fprob             = fprob ./ last.(fprob)
                  sstar             = [find(fprob[k] .> rand())[1] for k = 1:K]
                  for k = 1:K
                      particle_add(data[k][j, :], sstar[k], j, newParticle.clusters[k])
                  end
              end
        end
          logprob           = [calc_logprob(data[k][p, :], newParticle.clusters[k]) for k = 1:K]
          fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
          newParticle.logW  = newParticle.logW + sum(maximum.(logprob)) + sum(log.(last.(fprob)))
          fprob             = fprob ./ last.(fprob)
          sstar             = [find(fprob[k] .> rand())[1] for k = 1:K]

          for k = 1:K
              particle_add(data[k][p, :], sstar[k], p, newParticle.clusters[k])
          end
         # print(newParticle.clusters[1].c[1:p], "next", "\n")

      end
    end


    @inline function lG(p::Int64, particle::mdiParticle{K}, ::Void) where{K}
      return particle.logW
    end
    return SMCModel(M!, lG, n_obs, mdiParticle{K}, Void)
end
