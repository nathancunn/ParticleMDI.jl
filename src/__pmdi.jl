using DelimitedFiles
using Distributions
using IterTools
using NonUniformRandomVariateGeneration
using Printf
using Random
using SpecialFunctions
using Statistics
using StatsBase

"""
This is a test file. It runs exactly the same as pmdi but returns some internals.
Should not be used.
"""
function __pmdi(dataFiles, dataTypes, N::Int64, particles::Int64,
    ρ::Float64, iter::Int64;
     featureSelect::Union{String, Nothing} = nothing,
     dataNames = nothing)

    K       = length(dataFiles) # No. of datasets
    n_obs   = size(dataFiles[1], 1)

    # Set names if not specified
    if dataNames == nothing
        dataNames = ["K$i" for i in 1:K]
    end

    @assert length(dataTypes) == K "Number of datatypes not equal to number of datasets"
    @assert length(dataNames) == K "Number of data names not equal to number of datasets"
    @assert all(x->x==n_obs, [size(dataFiles[k])[1] for k = 1:K]) "Datasets don't have same number of observations. Each row must correspond to the same underlying observational unit across datasets."
    @assert (ρ < 1) && (ρ > 0) "ρ must be between 0 and 1"
    @assert (N <= n_obs) & (N > 1) "Number of clusters must be greater than 1 and not greater than the number of observations, suggest using floor(log(n)) = $(floor(Int64, log(n_obs)))"
    @assert particles > 1 "Conditional particle filter requires 2 or more particles"


    # Initialise the hyperparameters
    M = ones(Float64, K) .* 2 # Mass parameter
    γc = rand(Gamma(1.0 / N, 1), N, K) .+ eps(Float64) # Component weights
    Φ = K > 1 ? rand(Gamma(1, 0.2), Int64(K * (K - 1) * 0.5)) : zeros(1) # Dataset concordance measure

    # Initialise allocations randomly according to γc
    s = Matrix{Int64}(undef, n_obs, K)
    for k = 1:K
        s[:, k] = sampleCategorical(n_obs, γc[:, k])
    end

    # Get a matrix of all combinations of allocations
    c_combn = Matrix{Int64}(undef, N ^ K, K)
    for k in 1:K
        c_combn[:, K - k + 1] = div.(0:(N ^ K - 1), N ^ (K - k)) .% N .+ 1
    end

    # The corresponding gammas
    Γc = Matrix{Float64}(undef, N ^ K, K)
    log_γ = log.(γc)
    for k = 1:K
        Γc[:, k] = view(log.(γc[:, k]), c_combn[:, k])
    end


    # Which Φ value is activated by each of the above combinations
    Φ_index = K > 1 ? Matrix{Bool}(undef, N ^ K, Int64(K * (K - 1) / 2)) : fill(1, (N, 1))
    if K > 1
        i = 1
        for k1 in 1:(K - 1)
            for k2 in (k1 + 1):K
                Φ_index[:, i] = (c_combn[:, k1] .== c_combn[:, k2])
                i += 1
            end
        end
    end

    # Normalising constant
    Z = update_Z(Φ, Φ_index, Γc)
    v = update_v(n_obs, Z)

    # Particle weights
    logweight = zeros(Float64, particles)
    # Ancestor weights for ancestor sampling
    # ancestor_weights = zeros(Float64, particles)
    # Mutation weights
    logprob = Matrix{Float64}(undef, N * particles + 1, K)
    fprob = Vector{Float64}(undef, N)
    # Feature selection index
    featureFlag = [rand(Bool, size(dataFiles[k], 2)) for k in 1:K]
    if featureSelect == nothing
        for k in 1:K
            featureFlag[k] .= true
        end
    else
        featureNames = ["$(dataNames[k])_d$d" for k in 1:K for d in 1:size(dataFiles[k], 2)]
        writedlm(featureSelect, reshape(featureNames, 1, length(featureNames)), ',')
        featureFile = open(featureSelect, "a")
        writedlm(featureFile, [featureFlag...;]', ',')
    end


    # Feature select probabilities
    featureProb = [zeros(Float64, size(dataFiles[k], 2)) for k in 1:K]
    featureNull = [zeros(Float64, size(dataFiles[k], 2)) for k in 1:K]
    nullCluster = [dataTypes[k](dataFiles[k]) for k in 1:K]
    for k = 1:K
        for i = 1:n_obs
            cluster_add!(nullCluster[k], dataFiles[k][i, :], ones(Bool, size(dataFiles[k], 2)))
        end
        featureNull[k] = - calc_logmarginal(nullCluster[k])
    end

    # particle matches the cluster labels to the cluster IDs
    # particle = [fill(1, (N, particles)) for k in 1:K]
    particle = ones(Int, N, particles, K)
    particle_id = ones(Int, particles, K)
    fprob_dict = Matrix{Float64}(undef, N + 1, particles)
    fprob_done = Vector{Bool}(undef, particles)
    # logprob_particle = [Matrix{Float64}(undef, N, particles) for k in 1:K]
    logprob = Matrix{Float64}(undef, N * particles + 1, K)
    # logprob_particle = zeros(Float64, N, particles, K)
    logprob_particle = view(logprob, particle)

    # A vector containing all of the clusters
    clusters = [Vector{dataTypes[k]}(undef, N * particles + 1) for k in 1:K]
    # Keep track of how many copies of a cluster exists
    clusters_counts = zeros(Int, N * particles + 1, K)

    sstar_id = Matrix{Int64}(undef, particles, K)
    sstar = zeros(Int64, particles, n_obs, K)
    out = [map(x -> @sprintf("MassParameter_%d", x), 1:K);
               map((x, y) -> @sprintf("phi_%d_%d", x, y),
               calculate_Φ_lab(K)[:, 1],
               calculate_Φ_lab(K)[:, 2]);
               "ll";
               ["$(dataNames[k])_n$i" for k in 1:K for i in 1:n_obs]]
    out =  reshape(out, 1, length(out))
    # writedlm(outputFile, out, ',')
    # fileid = open(outputFile, "a")
    ll = 0
    ll1 = time_ns()
    # writedlm(fileid, [M; Φ; ll;  s[1:(n_obs * K)]]', ',')

    order_obs = collect(1:n_obs)
    n1 = floor(Int64, ρ * n_obs)

    n_operations = 0

    @inbounds for it in 1:iter
        fill!(clusters_counts, 0)
        clusters_counts[1, :] .= particles * N
        for i in eachindex(particle)
            particle[i] = 1
        end
        shuffle!(order_obs)

        # Update hyperparameters
        if K > 1
            update_Φ!(Φ, v, s, Φ_index, γc, K, Γc)
        end
        update_γ!(γc, Φ, v, M, s, Φ_index, c_combn, Γc, N, K)
        log_γ = log.(γc)

        Π = γc ./ sum(γc, dims = 1)
        Z = update_Z(Φ, Φ_index, Γc)
        v = update_v(n_obs, Z)
        update_M!(M, γc, K, N)


        for k = 1:K
            clusters[k][1] = dataTypes[k](dataFiles[k])
            clust_ids = Dict{Int, Int}()
            id = 2
            us = unique(s[order_obs[1:(n1 - 1)], k])
            for u in us
                clusters[k][id] = dataTypes[k](dataFiles[k])
                clusters_counts[id, k] = particles
                clusters_counts[1, k] -= particles
                clust_ids[u] = id
                particle[u, :, k] .= id
                id += 1
            end
            for i in order_obs[1:(n1 - 1)]
                # id = findall((in)(s[i, k]), us)[1] + 1
                id = clust_ids[s[i, k]]
                sstar[:, i, k] .= s[i, k]
                cluster_add!(clusters[k][id], dataFiles[k][i, :], featureFlag[k])
                n_operations += 1
            end
        end

        for i in order_obs[n1:n_obs]
            for k in 1:K
                fprob_done .= false
                particle_k = view(particle, :, :, k)
                logprob_particle = view(logprob, particle_k, k)
                obs = view(dataFiles[k], i, :)
                # logprob_particle_k = view(logprob_particle, :, :, k)
                sstar_id_k = view(sstar_id, :, k)
                Π_k = view(Π, :, k)
                for id in 1:maximum(particle_k)
                    logprob[id, k] = calc_logprob(obs, clusters[k][id], featureFlag[k])
                    n_operations += 1
                end
                # logprob_particle_k = logprob[particle_k, k]


                # Draw the new allocations
                for p in 1:particles
                    # fprob = logprob_particle[:, p]
                    # fprob_key = hash(sum(particle_k[:, p]))
                    # fprob_key = string(particle_k[:, p])

                    if fprob_done[particle_id[p, k]]
                        for n in 1:N
                        fprob[n] = fprob_dict[n, particle_id[p, k]]
                        end
                        logweight[p] += fprob_dict[end, particle_id[p, ]]
                    else
                        for n in 1:N
                             # fprob[n] = Π[n, k] * exp(fprob[n] - max_logprob)
                             fprob[n] = logprob_particle[n, p]
                         end
                         max_logprob = maximum(fprob)
                         for n in 1:N
                             fprob[n] -= max_logprob
                             fprob[n] = exp(fprob[n])
                             fprob[n] *= Π_k[n]
                        end
                        fprob = cumsum!(fprob, fprob)
                        for n in 1:N
                            fprob[n] = fprob[n] / fprob[N]
                        end
                        # fprob = fprob ./ maximum(fprob)
                        fprob_dict[1:(end - 1), particle_id[p, k]] = fprob
                        fprob_dict[end, particle_id[p]] =  log(sum(fprob)) + max_logprob
                        logweight[p] += log(sum(fprob)) + max_logprob
                        fprob_done[particle_id[p, k]] = true
                    end
                    # Set reference trajectory
                    if p != 1
                        new_s = 1
                        u = rand()
                        for i in 1:(N - 1)
                            if fprob[new_s] > u
                                break
                            else
                                new_s += 1
                            end
                        end
                        # new_s = sample(1:N, Weights(fprob))
                    else
                        new_s = s[i, k]
                    end
                    particle_id[p, k] *= (N * particles)
                    particle_id[p, k] += new_s
                    sstar_id_k[p] = particle[new_s, p, k]
                    sstar[p, i, k] = new_s
                end
                canonicalise_IDs!(view(particle_id, :, k))
                # Add observation to new cluster
                max_k = maximum(particle_k)
                for p in unique(sstar_id_k)
                    ncopies = count(x -> x == p, sstar_id_k)
                    # for s in sstar_id_k
                    #    if s == p
                    #        ncopies += 1
                    #    end
                    #end
                    # if wipedout(particle_k, sstar_id_k, p)
                    if ncopies == clusters_counts[p, k]
                        # If the origin cluster still exists somewhere
                        # Need to create a new cluster
                        # with obs added to it
                        id = p
                    else
                        id = max_k + 1
                        clusters_counts[p, k] -= ncopies
                        clusters_counts[id, k] = ncopies
                        # clusters[k][id] = deepcopy(clusters[k][p])
                        clusters[k][id] = copy_particle(clusters[k][p], dataFiles[k])
                        max_k += 1
                    end
                    cluster_add!(clusters[k][id], obs, featureFlag[k])
                    # n_operations += 1
                    if id !== p
                        for part in 1:particles
                            s_id = sstar[part, i, k]
                            if particle[s_id, part, k] == p
                                particle[s_id, part, k] = id
                            end
                        end
                    end
                end
            end
            if K > 1
                Φ_upweight!(logweight, sstar[:, i, :], K, Φ, particles)
            end

            # Resampling
            if calc_ESS(logweight) <=  0.5 * particles
                partstar = draw_partstar(logweight, particles)
                logweight .= 1.0
                for k in 1:K
                    particle[:, :, k] = particle[:, partstar, k]
                    particle_id[:, k] = particle_id[partstar, k]
                    canonicalise_IDs!(view(particle_id, :, k))
                    sstar[:, :, k] = sstar[partstar, :, k]
                    # Reset clusters_counts
                    clusters_counts[:, k] .= 0
                    # Renumber the clusters to ensure elements don't grow too large
                    particle_k = view(particle, :, :, k)
                    for (i, id) in enumerate(sort(unique(particle[:, :, k])))
                        if id !== i
                            for j in eachindex(particle_k)
                                if particle_k[j] == id
                                    particle_k[j] = i
                                end
                            end
                            # clusters[k][i] = deepcopy(clusters[k][id])
                            clusters[k][i] = copy_particle(clusters[k][id], dataFiles[k])
                        end
                        clusters_counts[i, k] = count(x -> x == i, particle_k)
                    end
                end
            end
        end

        # Select a single particle
        p_weights = similar(logweight)
        max_logweight = maximum(logweight)
        for i in eachindex(p_weights)
            p_weights[i] = exp(logweight[i] - max_logweight)
        end
        p_star = sample(1:particles, Weights(p_weights))

        # Feature selection
        ## Create a null particle with every obs in one cluster
        if featureSelect != nothing
            ## Compare this to the marginal likelihood for each cluster
            for k = 1:K
                featureProb[k] = featureNull[k] .+ 0
                occupiedClusters = unique(sstar[p_star, :, k])
                for clust in occupiedClusters
                    clust_members = findindices(sstar[p_star, :, k], clust)
                    clust_params = dataTypes[k](dataFiles[k])
                    for obs in clust_members
                        cluster_add!(clust_params, dataFiles[k][obs, :], ones(Bool, size(dataFiles[k], 2)))
                    end
                    featureProb[k] += calc_logmarginal(clust_params)
                end
                featureFlag[k] = (1 .- 1 ./ (exp.(featureProb[k] .+ 1))) .> rand(length(featureProb[k]))
                # featurePosterior[k] += featureFlag[k] ./ (iter + 1)
            end
        end

        logweight .= 1.0
        s[:] = sstar[p_star, :, :]
        # Match up labels across datasets
        align_labels!(s, Φ, γc, N, K)

        ll = (time_ns() - ll1) / 1.0e9
        #if it % thin == 0
        #    writedlm(fileid, [M; Φ; ll; s[1:(n_obs * K)]]', ',')
        #    if featureSelect != nothing
        #        writedlm(featureFile, [featureFlag...;]', ',')
        #    end
        #end
    end
    # close(fileid)
    if featureSelect != nothing
        close(featureFile)
    end
    return particle, clusters, clusters_counts, n_operations
end
