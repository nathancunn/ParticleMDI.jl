function update_v(n_obs::Int64, Z::Float64)
    return rand(Gamma(n_obs, 1 / Z))
end

function update_M!(M::Array, γ::Array, K::Int64, N::Int64)
    # Update the mass parameter
    prior = [2.0, 0.25]
    for k = 1:K
        @inbounds current_γ = γ[:, k]
        current_M = Float64(M[k])
        log_likelihood = - sum(logpdf.(Gamma(current_M / N, 1.0), current_γ))
        log_likelihood_0 = - sum(logpdf.(Gamma(prior[1], prior[2]), current_M))
        proposed_mass = (current_M + rand(Normal()) / 10)
        if proposed_mass <= 0.0
            alpha = 0.0
        else
            new_log_likelihood = - sum(logpdf.(Gamma(proposed_mass / N, 1), current_γ))
            new_log_likelihood_0 = - sum(logpdf(Gamma.(prior[1], prior[2]), proposed_mass))
            alpha = exp(-new_log_likelihood - new_log_likelihood_0 + log_likelihood + log_likelihood_0)
        end
        if rand() < alpha
            M[k] = proposed_mass
        end
    end
    return
end


function update_Z(Φ::Array, Φ_index::Array, Γ::Array)
    # Update the normalising constant
    Z = sum(exp.((Φ_index * (log.(Φ .+ 1)) + sum(Γ, dims = 2))))
    return Z
end

function calculate_likelihood(s::Array, Φ::Array, γ::Array, Z::Float64)
    likelihood = zeros(Float64, (size(s, 1)))
    Φ_log = similar(Φ)
    for i in eachindex(Φ)
        Φ_log[i] = log(1 + Φ[i])
    end
    for i in 1:size(s, 1)
        for k in 1:size(s, 2)
            likelihood[i] += log(γ[s[i, k], k])
        end
        if size(s, 2) > 1
            ϕ = 1
            for k1 in 1:(size(s, 2) - 1)
                for k2 in (k1 + 1):(size(s, 2))
                    likelihood .+= Φ_log[ϕ] * s[i, k1] == s[i, k2]
                    ϕ += 1
                end
            end
        end
    end
    return sum(exp.(likelihood) ./ Z)
end

function update_γ!(γ::Array, Φ::Array, v::Float64, M, s::Array, Φ_index::Array, γ_combn::Array, Γ::Array, N::Int64, K::Int64)
    β_0 = 1.0
    Φ_log = log.(Φ .+ 1)
    α_star = Matrix{Float64}(undef, N, K)
    for k = 1:K
        for n = 1:N
            # @inbounds α_star[n, k] = α_0 + sum(s_k .== n)
            # @inbounds α_star[n, k] = M[k] / N + sum(s[:, k] .== n)
            @inbounds α_star[n, k] = M[k] / N + countn(s[:, k], n)
        end
    end
    norm_temp = Φ_index * Φ_log + sum(Γ, dims = 2)
    norm_temp = exp.(norm_temp)
    for k = 1:K
        @inbounds γ_combn_k = γ_combn[:, k]
         for n = 1:N
            pertinent_rows = findZindices(k, K, n, N)
            old_γ = γ[n, k] + 0.0
            # @inbounds β_star = β_0 + v * sum(exp.(Φ_index[pertinent_rows, :] * Φ_log + sum(Γ[pertinent_rows, :], 2))) / γ[n, k]
            @inbounds β_star = β_0 + v * sum((norm_temp[pertinent_rows])) / γ[n, k]
            @inbounds γ[n, k] = rand(Gamma(α_star[n, k], 1 / β_star)) + eps(Float64)
            @inbounds norm_temp[pertinent_rows] .*= γ[n, k] / old_γ
        end
    end
    return
end


function update_Φ!(Φ, v::Float64, s, Φ_index, γ, K::Int64, Γ)
    # Prior parameters
    if K == 1
        return
    else
        # Prior parameters
        α_0 = 1.0
        β_0 = 0.2
        Φ_lab = calculate_Φ_lab(K)
        Φ_log = log.(Φ .+ 1)
        norm_temp = Φ_index * Φ_log + sum(Γ, dims = 2)
        @fastmath norm_temp = exp.(norm_temp)
        for i in 1:length(Φ)
            # Get relevant allocations
            @inbounds current_allocations = s[:, Φ_lab[i, :]]
            @inbounds Φ_current = Φ[i] + 0.0
            @inbounds n_agree = sum(current_allocations[:, 1] .== current_allocations[:, 2])
            # Get relevant terms in the normalisation constant Terms that include the current phi
            @inbounds pertinent_rows = findall(Φ_index[:, i])
            @inbounds β_star = β_0 + v * sum(norm_temp[pertinent_rows, :]) / (1 + Φ_current)
            #weights = cumsum(log.((0:n_agree) .+ α_0))  # Add initial 1 to account for zero case
            weights = lgamma.((0:n_agree) .+ α_0)
            # weights = gamma.((0:n_agree) .+ α_0)
            weights += logpdf.(Binomial(n_agree, 0.5), 0:n_agree)
            # weights += log.(binomial.(n_agree, 0:n_agree))
            # weights .*= (binomial.(n_agree, 0:n_agree))
            # ADDING BETA BELOW FIXES THIS SOMEHOW
            weights -= (1:(n_agree + 1)) .* log(β_star)
            ## DOING THIS ALL LIKE A CUMSUM
            # weights ./= β_star .^ (0:n_agree)
            # weights = log.(weights)
            # if length(weights) != (n_agree + 1)
            # println(exp.(weights .- maximum(weights)))
            #    println((ones(1 +  n_agree) .* log(β_star)))
            #end

            α_star = α_0 + sample(0:n_agree, Weights(exp.(weights .- maximum(weights))))
            @inbounds Φ[i] = rand(Gamma(α_star, 1 / β_star)) + eps(Float64)
            # @inbounds Φ_log[i] = log(Φ[i] + 1)
            # Update the normalising constant values to account for this update
            @inbounds norm_temp[pertinent_rows, :] .*= (1 + Φ[i]) / (1 + Φ_current)

        end
    end
    return
end
