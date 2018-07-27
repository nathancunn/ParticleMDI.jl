# Implement particle cluster analysis for univariate Gaussian data
## Load the packages
using SequentialMonteCarlo
using StaticArrays

include("particles.jl")


function pMDImodel(data, dataTypes, N, particles, s, ρ, nThreads, fullOutput, essThreshold)
    K       = length(data) # No. of datasets
    n_obs   = size(data[1])[1]
    d       = [size(data[k])[2] for k = 1:K]
    @assert length(dataTypes) == K
    @assert all(x->x==n_obs, [size(data[k])[1] for k = 1:K])
    # The mutation function
    function M!(newParticle::mdiParticle{K}, rng::SMCRNG, p::Int64,
      particle::mdiParticle{K}, scratch::mdiScratch{K}) where {K}
      i_1 = Int64(floor(ρ * n_obs))
      K = length(data)
      n_obs   = size(data[1])[1]


      if p <= i_1
          if p == 1
              for k = 1:K
                  newParticle.clusters[k] = dataTypes[k]{n_obs, d[k], N}()
              end
          end

        #      for i = 1:i_1
        #          particle_add(data[k][i, :], Int64(s[k][i]), i, newParticle.clusters[k])
        #      end
         # end
          # Then add this first free observation to a cluster
          #logprob           = [calc_logprob(data[k][1 + i_1, :], newParticle.clusters[k]) for k = 1:K]
          #fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
          #newParticle.logW  = newParticle.logW + sum(maximum.(logprob)) + sum(log.(last.(fprob)))
          #fprob             = fprob ./ last.(fprob)
          #newParticle.c     = [find(fprob[k] .> rand())[1] for k = 1:K]
          newParticle.c      = [s[k][p] for k = 1:K]


          for k = 1:K
              particle_add!(data[k][1 + i_1, :], newParticle.c[k], 1 + i_1, newParticle.clusters[k])
          end
          # Set up scratch space
          # All particles have same ID initially
          scratch.currentID = 1
          scratch.newID     = 1
          scratch.logprob   = [zeros(N) for k in 1:K]
          scratch.iter      = 1
          newParticle.ID    = 1
      else
          if smcio.resample[p - 1] == true
              # Reset particle weights
              newParticle.logW = 0
        end
        if scratch.iter == p & scratch.currentID != particle.ID
            scratch.logprob         = [calc_logprob(data[k][p, :], particle.clusters[k]) for k = 1:K]
            scratch.newID           += 1
            scratch.currentID       = particle.ID
            # error("It's haaaapppenninggg")
        elseif scratch.iter != p
            #print("Scratch not equal p ", scratch.iter, " ", p, " \n")
            scratch.iter        = p
            scratch.newID       = 1
            scratch.currentID   = particle.ID
            scratch.logprob     = [calc_logprob(data[k][p, :], particle.clusters[k]) for k = 1:K]
        end
          logprob           = scratch.logprob
          newParticle.ID    = scratch.newID
          fprob             = [cumsum(exp.(logprob[k] .- maximum(logprob[k]))) for k = 1:K]
          newParticle.logW  = newParticle.logW + sum(maximum.(logprob)) + sum(log.(last.(fprob)))
          fprob             = fprob ./ last.(fprob)
          newParticle.c     = [find(fprob[k] .> rand())[1] for k = 1:K]
          for k = 1:K
              particle_add!(data[k][p, :], newParticle.c[k], p, newParticle.clusters[k])
          end

          for k = 1:K
              newParticle.ID = Int64(newParticle.ID * N + newParticle.c[k])
          end
      end
    end

    # The particle weight
    function lG(p::Int64, particle::mdiParticle{K}, scratch::mdiScratch{K}) where {K}
      return particle.logW
    end

    model = SMCModel(M!, lG, n_obs, mdiParticle{K}, mdiScratch{K})
    return model

end
