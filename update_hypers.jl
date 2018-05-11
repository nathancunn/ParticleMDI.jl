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
            @inbounds M[k] = proposed_mass
        end
    end
    return
end


function update_Z(Φ::Array, Φ_index::Array, Γ::Array)
    # Update the normalising constant
    Z = Float64(sum(exp.(sum(Φ_index .* transpose(log.(Φ .+ 1)), 2) .+ sum(Γ, 2))))
    return(Z)
end

function update_γ!(γ::Array, Φ::Array, v::Float64, s::Array, Φ_index::Array, γ_combn::Array, Γ::Array, N::Int64, K::Int64)
    α_0 = 1.0 / N
    β_0 = 1.0
    Φ_log = log.(Φ .+ 1)
    α_star = Matrix{Float64}(N, K)
    for k = 1:K
        s_k = s[:, k]
        for n = 1:N
            @inbounds α_star[n, k] = α_0 + sum(s_k .== n)
        end
    end
    for k = 1:K
        @inbounds γ_combn_k = γ_combn[:, k]
        for n = 1:N
            pertinent_rows = γ_combn_k .== n
            @inbounds β_star = β_0 + v * sum(exp.(Φ_index[pertinent_rows, :] * Φ_log + sum(Γ[pertinent_rows, :], 2))) / γ[n, k]
            @inbounds γ[n, k] = rand(Gamma(α_star[n, k], 1 / β_star)) + eps(Float64)
        end
    end
    return
end


function update_Φ!(Φ, v::Float64, s, Φ_index, γ, K::Int64, Γ)
    # Prior parameters
    α_0 = 1
    β_0 = 5
    if K == 1
        return
    else
        # Prior parameters
        α_0 = 1
        β_0 = 0.2
        Φ_lab = calculate_Φ_lab(K)
        for i in 1:length(Φ)
            # Get relevant allocations
            @inbounds current_allocations = s[:, Φ_lab[i, :]]
            @inbounds Φ_current = Φ[i]
            @inbounds n_agree = sum(current_allocations[:, 1] .== current_allocations[:, 2])
            # Get relevant terms in the normalisation constant Terms that include the current phi
            @inbounds pertinent_rows = Φ_index[:, i] .== 1
            @inbounds β_star = β_0 + v * sum(exp.(Φ_index[pertinent_rows, :] * log.(1 + Φ) + sum(Γ[pertinent_rows, :], 2))) / Φ_current
            weights = cumsum(log.(α_0 .+ [1, 1:n_agree...]))  # Add initial 1 to account for zero case
            weights += logpdf.(Binomial(n_agree, 0.5), 0:n_agree)
            weights -= cumsum((ones(1 +  n_agree) .* log(β_star)))
            α_star = α_0 + sample(1:length(weights), Weights(exp.(weights .- maximum(weights)))) - 1
            @inbounds Φ[i] = rand(Gamma(α_star, 1 / β_star)) + eps(Float64)
        end
    end
    return
end
