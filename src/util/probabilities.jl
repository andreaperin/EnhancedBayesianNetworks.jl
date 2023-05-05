function verify_probabilities(states::Dict{Symbol,<:Real})
    any(values(states) .< 0.0) && error("Probabilites must be nonnegative")
    any(values(states) .> 1.0) && error("Probabilites must be less or equal to 1.0")
    sum(values(states)) != 1 && error("Probabilites must sum up to 1.0")
end