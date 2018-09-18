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


function draw_sstar!(sstar::Array, logprob, particle::Array, particle_IDs::Array, Π::Array, K::Int64, N::Int64, ancestor_weights::Vector, logweight::Vector, s)
    fprob = Vector{Float64}(N)
    for k = 1:K
        for p = 1:maximum(particle_IDs[:, k])
            @inbounds particle_flag = findindices(particle_IDs[:, k], p)
            @inbounds max_p = maximum(logprob[:, p, k])
            for n = 1:N
                @fastmath @inbounds fprob[n] = Π[n, k] * exp(logprob[n, p, k] - max_p)
            end
            # Draw sstar
            # @inbounds @fastmath sstar[particle_flag, k] = sample(Vector{Int64}(1:N), Weights(fprob), length(particle_flag))
            @inbounds sstar[particle_flag, k] = sampleCategorical(length(particle_flag), fprob)
            # Update ancestor weights
            for p_flag in particle_flag
                @inbounds @fastmath ancestor_weights[p_flag] += Base.Math.JuliaLibm.log(fprob[s[k]] + eps(Float64))
                # Update logweight
                @inbounds @fastmath logweight[p_flag] += Base.Math.JuliaLibm.log(sum(fprob)) + max_p
            end
        end
    end
    return
end

function draw_sstar_p!(sstar, logprob, Π::Array, N::Int64, ancestor_weights::Vector, logweight, s)
    fprob = Vector{Float64}(N)
    # @inbounds particle_flag = findindices(particle_IDs, p)
    @inbounds max_p = maximum(logprob)
    for n = 1:N
        @fastmath @inbounds fprob[n] = Π[n] * exp(logprob[n] - max_p)
    end
    # Draw sstar
    @inbounds sstar = sampleCategorical(length(sstar), fprob)
    # Update ancestor weights
    #for p_flag in particle_flag
    @inbounds @fastmath ancestor_weights += Base.Math.JuliaLibm.log(fprob[s] + eps(Float64))
        # Update logweight
        @inbounds @fastmath logweight .+= Base.Math.JuliaLibm.log(sum(fprob)) + max_p
    #end
    return (sstar, logweight)
end

@inline function update_particleIDs(particle_IDs, sstar, particles, N)
    return ID_to_canonical!(particle_IDs * (particles * N) + sstar)
end

@inline function update_particleIDs!(particle_IDs, sstar, particles, N)
    for k in 1:size(particle_IDs, 2)
        for p in 1:size(particle_IDs, 1)
            particle_IDs[p, k] = particle_IDs[p, k] * (particles * N) + sstar[p, k]
        end
    end
    ID_to_canonical!(particle_IDs)
end

function ID_to_canonical!(x)
    for (i, xi) in enumerate(x)
        if xi > 0
            for k in eachindex(x)
                if x[k] == xi
                    x[k] = - i
                end
            end
        end
    end
    x = - x
end


@inline function calc_ESS(logweight)
    ESSnum = 0.0
    ESSdenom = 0.0
    max_l = maximum(logweight)
    for l in logweight
        num = Base.Math.JuliaLibm.exp(l - max_l)
        ESSnum += num
        ESSdenom += num ^ 2
    end
    # return sum(exp.(logweight .- maximum(logweight))) .^ 2 / sum(exp.(logweight .- maximum(logweight)) .^ 2)
    return (ESSnum ^ 2) / ESSdenom
end

function draw_partstar(logweight, particles)
    u = rand() / particles
    pprob = cumsum(exp.(logweight .- maximum(logweight)))
    partstar = zeros(Int64, particles)
    i = 0
    for p = 1:particles
        while pprob[p] / last(pprob) >= u
            u += 1 / particles
            i += 1
            partstar[i] = p
        end
    end
    partstar[1] = 1
    return partstar
end


function Φ_upweight!(logweight, sstar, K::Int64, Φ, particles)
    if K == 1
        return
    else
        Φ_lab = calculate_Φ_lab(K)
        for i in 1:Int64((K * (K - 1) * 0.5))
            Φ_log = log(1 + Φ[i])
            for p in 1:particles
                logweight[p] += (sstar[p, Φ_lab[i, 1]] == sstar[p, Φ_lab[i, 2]]) * Φ_log
            end
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


function ID_match(particle_IDs, new_particle_IDs, particles::Int64)
    # Find the minimum new_particle_ID corresponding
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
    for (i, a) in enumerate(A)
        if a == b
            push!(out, i)
        end
    end
    return out
end

function countn(A, b)
    out = 0
    for a in A
        if a == b
            out += 1
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


function wipedout(v1, v2, x)
    count1 = 0
    for i in eachindex(v1)
        if v1 == x
            count1 += 1
        end
    end
    for i in eachindex(v2)
        if v2 == x
            count1 -= 1
        end
        if count1 <= 0
            return false
        end
    end
    return true
end
