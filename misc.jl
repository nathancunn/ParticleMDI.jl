function calculate_Φ_lab(K::Int64)
    Φ_lab = K > 1 ? Matrix{Int64}(Int64(K * (K - 1) / 2), 2) : ones(Int64, 1, 2)
    if K > 1
        i = 1
        for k1 in 1:(K - 1)
            for k2 in (k1 + 1):K
                Φ_lab[i, :] = [k1, k2]
                i += 1
            end
        end
    end
    return Φ_lab
end

# Not needed anymore
function calc_fprob!(fprob::Array, logprob, particle_IDs, K, Π)
    @simd for k = 1:K
        @simd for p in unique(particle_IDs[:, k])
            fprob[:, p, k] = Π[:, k] .* exp.(logprob[:, p, k] .- maximum(logprob[:, p, k]))
        end
    end
    return
end

function update_logweight!(logweight::Array, particle::Array, particle_IDs::Array, Π::Array, K::Int64, N::Int64)
    @simd for k = 1:K
        # π_k = Π[:, k]
        # ids_k = particle_IDs[:, k]
        for p in 1:maximum(particle_IDs[:, k])
            @inbounds fprob = Π[:, k] .* exp.(particle[k][p].ζ .- maximum(particle[k][p].ζ))
            @inbounds logweight[findindices(particle_IDs[:, k], p)] .+= log(sum(fprob)) + maximum(particle[k][p].ζ)
        end
    end
    return
end


function draw_sstar!(sstar::Array, logprob, particle::Array, particle_IDs::Array, Π::Array, K::Int64, N::Int64, ancestor_weights::Vector, logweight::Vector, s)
    fprob = Vector{Float64}(N)
    for k = 1:K
        for p = 1:maximum(particle_IDs[:, k])
            particle_flag = findindices(particle_IDs[:, k], p)
            max_p = maximum(logprob[k][:, p])
            for n = 1:N
                @fastmath @inbounds fprob[n] = Π[n, k] * exp(logprob[k][n, p] - max_p)
            end
            # Draw sstar
            @inbounds @fastmath sstar[particle_flag, k] = sample(Vector{Int64}(1:N), Weights(fprob), length(particle_flag))
            # Update ancestor weights
            for p_flag in particle_flag
                @inbounds @fastmath ancestor_weights[p_flag] += log(fprob[s[k]] + eps(Float64))
                # Update logweight
                # @inbounds @fastmath logweight[p_flag] += log(sum(fprob)) + maximum(particle[k][p].ζ)
                @inbounds @fastmath logweight[p_flag] += log(sum(fprob)) + max_p
            end
        end
        # ancestor_weights += logweight
    end
    return
end


function update_particleIDs(particle_IDs, sstar, K, particles, N)
    return ID_to_canonical!(particle_IDs * (particles * N) + sstar)
end

function ID_to_canonical!(x)
    @simd for k in 1:size(x, 2)
        @inbounds u = unique(x[:, k])
        @inbounds x[:, k] = indexin(x[:, k], u)
    end
    return x
end


function calc_ESS(logweight)
    return sum(exp.(logweight .- maximum(logweight))) .^ 2 / sum(exp.(logweight .- maximum(logweight)) .^ 2)
end

function draw_partstar(logweight, particles)
    u = rand() / particles
    pprob = cumsum(exp.(logweight .- maximum(logweight)))
    partstar = zeros(Int64, particles)
    i = Int64(0)
    for p = 1:particles
        while pprob[p] / last(pprob) >= u
            u += 1 / particles
            i += 1
            partstar[i] = p
        end
    end
    return partstar
end


function Φ_upweight!(logweight::Array, sstar::Array, K::Int64, Φ::Array)
    if K == 1
        return logweight
    else
        Φ_lab = calculate_Φ_lab(K)
        for i = 1:Int64((K * (K - 1) * 0.5))
            logweight .+= (sstar[:, Φ_lab[i, 1]] .== sstar[:, Φ_lab[i, 2]]) .* transpose(log(1 + Φ[i]))
        end
    end
    return
end


function Φ_involvement(K::Int64, k::Int64)
    Φ_lab = calculate_Φ_lab(K)
    Φ_involved = (Φ_lab[:, 1] .== k) .| (Φ_lab[:, 2] .== k)
    return Φ_involved
end


function align_labels!(s::Array, Φ::Array, γ::Array, N::Int64, K::Int64)
    if K == 1
        return
    else
        Φ_lab = calculate_Φ_lab(K)
        Φ_log = log.(1 + Φ)
        unique_s = unique(s)
        #if length(unique_s) > 1
            for k = 1:K
                unique_sk = unique(s[:, k])
                relevant_Φs = Φ_log[(Φ_lab[:, 1] .== k) .| (Φ_lab[:, 2] .== k)]
                # for i in 1:(length(unique_sk))
                for label in unique_sk
                    # new_label = sample(setdiff(unique_s, label))
                    for new_label in setdiff(unique_s, label)
                        label_ind = findindices(s[:, k], label)
                        new_label_ind = findindices(s[:, k], new_label)

                        label_rows      = s[label_ind, setdiff(1:K, k)]
                        new_label_rows  = s[new_label_ind, setdiff(1:K, k)]
                        log_phi_sum     = sum(sum(label_rows .== label, 1) .* relevant_Φs + sum(new_label_rows .== new_label, 1) .* relevant_Φs)
                        log_phi_sum_swap = sum(sum(label_rows .== new_label, 1) .* relevant_Φs + sum(new_label_rows .== label, 1) .* relevant_Φs)

                        accept = min(1, exp(log_phi_sum_swap - log_phi_sum))

                        if rand() < accept
                            # display(s)
                            s[label_ind, k]        .= new_label
                            s[new_label_ind, k]    .= label
                            # display(s)
                            γ_temp = γ[new_label, k]
                            γ[new_label, k] = γ[label, k]
                            γ[label, k] = γ_temp
                        end
                    end
                end
            end
        # end
    end
end


function ancestor_sampling!(logweight, particle_IDs, particle, particles)
    # This step alters the ID of the reference trajectory
    # Porabilistically pick the particle most likely to
    # mutate to the reference path in this step
    # and assign that ID to the reference trajectory
    # Ref particle mutates to reference path but its history
    # changes
    ancestor_weights = zeros(Float64, particles) + logweight
    for k = 1:K
        for p = 1:maximum(particle_IDs[:, k])
            # println(log((Π[:, k] .* exp.(particle[k][p].ζ .- maximum(particle[k][p].ζ)))[s[i, k]]) + eps(Float64))
            ancestor_weights[findindices(particle_IDs[:, k], p)] .+= log.((Π[:, k] .* exp.(particle[k][p].ζ .- maximum(particle[k][p].ζ)))[s[i, k]] + eps(Float64))
        end
    end
    ancestor_index = sample(1:particles, Weights(exp.(ancestor_weights .- maximum(ancestor_weights))))
    #for k = 1:K
    #         particle_IDs[1, k] .= particle_IDs[ancestor_index, k]
    #    end
    return ancestor_weights

end

function ancestor_sampling_2!(logweight, particle_IDs, particle, particles)
    # This step alters the ID of the reference trajectory
    # Porabilistically pick the particle most likely to
    # mutate to the reference path in this step
    # and assign that ID to the reference trajectory
    # Ref particle mutates to reference path but its history
    # changes
    ancestor_weights = zeros(Float64, particles) + logweight
    for k = 1:K
        for p = 1:particles
            ancestor_weights[p] += log((Π[s[i, k], k] * exp(particle[k][p].ζ[s[i, k]] - maximum(particle[k][p].ζ))) +eps(Float64))
        end
    end
    max_ancestor_weight = maximum(ancestor_weights)
    for p = 1:particles
        ancestor_weights[p] = exp(ancestor_weights[p] - max_ancestor_weight)
    end
    ancestor_index = sample(1:particles, Weights(ancestor_weights))
    #for k = 1:K
    #         particle_IDs[1, k] .= particle_IDs[ancestor_index, k]
    #    end
    return ancestor_weights

end


function ID_match(particle_IDs, new_particle_IDs, particles::Int64)
    # Find the maximum new_particle_ID corresponding
    # to a particular particle_ID
    out = Array{Int64}(particles)
    for u in unique(particle_IDs)
        out[u] = minimum(new_particle_IDs[findindices(particle_IDs, u)])
    end
    return out
end


function findindex(A, b)
    # Find first occurrence of b in A
    for (i,a) in enumerate(A)
        if a == b
            return i
        end
    end
end

function findindices(A, b)
    # Find all occurrences of b in A
    # Specifically for finding occurrences of gamma
    out = Int64[]
    count = 1
    for (i, a) in enumerate(A)
        if a == b
            push!(out, i)
        end
    end
    return out
end


function findgammas(n::Int64, k::Int64, N::Int64, K::Int64)
    pertinent_rows = Vector{Int64}(N ^ (K - 1))
    count = 1
         for j in collect(1:(N ^ (k - 1)))
             for i in collect(0:(N ^ k):(N ^ K - 1))
             pertinent_rows[count] = Int64(i + j + n - 1)
             count += 1
         end
    end
    return pertinent_rows
end
