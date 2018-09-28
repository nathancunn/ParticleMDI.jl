function calculate_Φ_lab(K::Int64)
    Φ_lab = K > 1 ? Matrix{Int64}(undef, Int64(K * (K - 1) / 2), 2) : fill(1, (1, 2))
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

@inline function calc_ESS(logweight)
    ESSnum = 0.0
    ESSdenom = 0.0
    max_l = maximum(logweight)
    for l in logweight
        num = exp(l - max_l)
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
    # Systematic resampling not ideal for CSMC
    # Particle indices are sorted, so can't just set
    # the first as reference trajectory
    # Instead: shuffle, replace first, sort
    shuffle!(partstar)
    partstar[1] = 1
    sort!(partstar)
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

function align_labels!(s::Array, Φ::Array, γ::Array, N::Int64, K::Int64)
    if K == 1
        return
    else
        Φ_lab = calculate_Φ_lab(K)
        Φ_log = log.(Φ .+ 1)
        unique_s = unique(s)
        #if length(unique_s) > 1
            for k = 1:K
                unique_sk = unique(s[:, k])
                relevant_Φs = Φ_log[(Φ_lab[:, 1] .== k) .| (Φ_lab[:, 2] .== k)]
                # for i in 1:(length(unique_sk))
                for label in unique_sk
                    new_label = sample(setdiff(unique_s, label))
                    # for new_label in setdiff(unique_s, label)
                        # label_ind = findindices(s[:, k], label)
                        label_ind = findall(s[:, k] .== label)
                        # new_label_ind = findindices(s[:, k], new_label)
                        new_label_ind = findall(s[:, k] .== new_label)
                        label_rows      = s[label_ind, setdiff(1:K, k)]
                        new_label_rows  = s[new_label_ind, setdiff(1:K, k)]
                        log_phi_sum     = sum(sum(label_rows .== label, dims = 1) .* relevant_Φs + sum(new_label_rows .== new_label, dims = 1) .* relevant_Φs)
                        log_phi_sum_swap = sum(sum(label_rows .== new_label, dims = 1) .* relevant_Φs + sum(new_label_rows .== label, dims = 1) .* relevant_Φs)

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
                # end
            end
        # end
    end
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

function findZindices(k, K, n, N)
    # Instead of searching where the γc_combn = a
    # by searching the vector
    # We can know this in advance
    # This way is >10x faster for larger problems
    out = zeros(Int, N ^ (K - 1))
    start = (n - 1) * N ^ (k - 1) + 1
    ind = 1
    for i in 1:(N ^ (K - k))
        for j in start:(start - 1 + N ^ (k - 1))
            out[ind] = j
            ind += 1
        end
        start += N ^ k
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
