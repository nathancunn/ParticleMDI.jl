using CSVFiles
using Distributions
using Iterators
using StatsBase

include("gaussian_cluster.jl")
include("update_hypers.jl")
include("misc.jl")


function pmdi(dataFiles, dataTypes, N::Int64, particles::Int64,
    ρ::Float64, iter::Int64, outputFile::String, initialise::Bool, output_freq::Int64)
    K       = length(dataFiles) # No. of datasets
    n_obs   = size(dataFiles[1])[1]
    d       = [size(dataFiles[k])[2] for k = 1:K]
    @assert length(dataTypes) == K
    @assert all(x->x==n_obs, [size(dataFiles[k])[1] for k = 1:K])

    # Normalise any Gaussian datasets
    ## Doing this as it's done in the MATLAB version
    ## Prefer to leave normalisation to user
    for k = 1:K
        if dataTypes[k] == "gaussianCluster"
            gaussian_normalise!(dataFiles[k])
        end
    end

    # Initialise the hyperparameters
    M = ones(K) * 2 # Mass parameter
    γ = rand(Gamma(2 / N, 1), N, K) .+ eps(Float64) # Component weights
    Φ = K > 1 ? rand(Gamma(1, 5), Int64(K * (K - 1) *0.5)) : zeros(1) # Dataset concordance measure

    # Initialise the allocations randomly according to γ
    s = Matrix{Int64}(zeros(n_obs, K))
    for k = 1:K
        s[:, k] = sample(1:N, Weights(γ[:, k]), n_obs)
    end

    # Get a matrix of all combinations of gammas
    γ_combn = Matrix{Int64}(N ^ K, K)
    for (i, p) in enumerate(product([1:N for k = 1:K]...))
        γ_combn[i, :] = collect(p)
    end

    # Which Φ value is activated by each of the above combinations
    Φ_index = K > 1 ? Matrix{Int64}(N ^ K, Int64(K * (K - 1) / 2)) : ones(Matrix{Bool}(N, 1), Int64)
    if K > 1
        i = 1
        for k1 in 1:(K - 1)
            for k2 in (k1 + 1):K
                Φ_index[:, i] = (γ_combn[:, k1] .== γ_combn[:, k2])
                i += 1
            end
        end
    end

    Γ = Matrix{Float64}(N ^ K, K) # The actual gamma values for the gamma combinations
    for k = 1:K
        Γ[:, k] = log.(γ[:, k][γ_combn[:, k]])
    end
    Z = update_Z(Float64(0), Φ, Φ_index, Γ)

    logweight = zeros(Float64, particles)
    sstar = zeros(Matrix{Int64}(particles, K), Int64)

    # Initialise the particles
    particle = [Vector{dataTypes[k]{n_obs, d[k], N}}(particles) for k = 1:K]
    particle_IDs = ones(Matrix{Int64}(particles, K), Int64)
    for k = 1:K
        particle[k][1] = dataTypes[k]{n_obs, d[k], N}()
    end

    # Save information to file
    if K > 1
        out = vcat(map(x -> @sprintf("mass_%d", x), 1:K), map((x, y) -> @sprintf("phi_%d_%d", x, y), calculate_Φ_lab(K)[:, 1], calculate_Φ_lab(K)[:, 2]), map((x, y) -> @sprintf("K%d_n%d", x, y), repeat(1:K, inner = n_obs), repeat(1:n_obs, outer = K)))
    else
        out = vcat(map(x -> @sprintf("mass_%d", x), 1:K), map((x, y) -> @sprintf("K%d_n%d", x, y), repeat(1:K, inner = n_obs), repeat(1:n_obs, outer = K)))
    end
    out =  reshape(out, 1, length(out))
    writecsv(outputFile, out)

    fileid = open(outputFile, "a")

    for it in 1:iter
        # Reset particles and IDs
        for k = 1:K
            particle_reset!(particle[k][1])
        end
        particle_IDs .= Int64(1)

        v = update_v!(n_obs, Z)
        update_M!(M, γ, K, N)

        Π = γ ./ sum(γ, 1)

        order_obs = randperm(n_obs)

        for k = 1:K
            for i in order_obs[1:Int64(floor(ρ * n_obs))]
                particle[k][1] = particle_add(dataFiles[k][i, :], i, s[i, k], particle[k][1])
            end
        end

        ## The conditional particle filter
        for i in order_obs[Int64(floor(ρ * n_obs) + 1):n_obs]
            for k = 1:K
                # for p in unique(particle_IDs[:, k])
                obs = dataFiles[k][i, :]
                for p in 1:maximum(particle_IDs[:, k])
                    calc_logprob!(obs, particle[k][p])
                end
            end

            # calc_fprob!(fprob, logprob, particle_IDs, K, Π)
            update_logweight!(logweight, particle, particle_IDs, Π, K, N)

            draw_sstar!(sstar, particle, particle_IDs, Π, K, N)
            # Set first particle to reference trajectory
            for k = 1:K
                sstar[1, k] = s[i, k]
            end

            #########################
            # Ancestor sampling step
            ########################
            # Take the fprob from each data at the ref trajectory's value
            # Add this to the particles current weight
            # Multinomially select index according to this
            ancestor_weights = zeros(Float64, particles) + logweight
            for k = 1:K
                for p = unique(particle_IDs[:, k])
                    # println(log((Π[:, k] .* exp.(particle[k][p].ζ .- maximum(particle[k][p].ζ)))[s[i, k]]) + eps(Float64))
                    ancestor_weights[findin(p, particle_IDs)] += log.((Π[:, k] .* exp.(particle[k][p].ζ .- maximum(particle[k][p].ζ)))[s[i, k]] + eps(Float64))
                end
            end
            ancestor_index = sample(1:particles, Weights(ancestor_weights), 1)[1]
            for k = 1:K
                particle_IDs[1, k] .= particle_IDs[ancestor_index, k]
            end

            # Update the particle IDs
            new_particle_IDs = update_particleIDs!(particle_IDs, sstar, K, particles, N)


            # Add observation to cluster
            for k = 1:K
                obs = dataFiles[k][i, :]
                # Doing this in reverse order means we don't need to copy the particles
                # Selecting the max value is quicker than checking for unique IDs
                # If the largest ID is n, then all IDs 1:n exist
                # If new_ID_1 > new_ID_2, then ID_2 ≧ ID_1
                for p in maximum(new_particle_IDs[:, k]):-1:1
                    p_ind = findin(new_particle_IDs[:, k], p)[1]
                    particle[k][p] = particle_add(obs, i, sstar[p_ind, k], particle[k][particle_IDs[p_ind, k]])
                end
            end

            particle_IDs = deepcopy(new_particle_IDs)
             Φ_upweight!(logweight, sstar, K, Φ)


            if calc_ESS(logweight) <= 0.5 * particles
                partstar = draw_partstar(logweight, particles)
                partstar[1] = 1
                logweight .= 1.0

                for k = 1:K
                        particle_IDs[:, k] = particle_IDs[partstar, k]
                end
            end
        end
        # Select a single particle
        p_star = sample(1:particles, Weights(logweight))
            for k = 1:K
                s[:, k] = particle[k][particle_IDs[:, k][p_star]].c
            end
            # Match up labels across datasets
            align_labels!(s, Φ, γ, N, K)
            update_Φ!(Φ, v, s, Φ_index, γ, K, Γ)
            update_γ!(γ, Φ, v, s, Φ_index, γ_combn, Γ, N, K)
            for k = 1:K
                Γ[:, k] = log.(γ[γ_combn[:, k], k])
            end
            Z = update_Z(Float64(0), Φ, Φ_index, Γ)
            if K > 1
                writecsv(fileid, [M; Φ; s[1:(n_obs * K)]]')
            else
                writecsv(fileid, [M; s[1:(n_obs * K)]]')
            end
        end
        close(fileid)
        return s
end
