using UncertaintyQuantification

function probabilities_of_events(
    models::Union{Array{<:UQModel},UQModel},
    performances::Dict{Symbol,Function},
    inputs::Union{Array{<:UQInput},UQInput},
    sim::AbstractMonteCarlo,
)
    samples = UncertaintyQuantification.sample(inputs, sim)
    evaluate!(models, samples)

    # Probabilities of failures
    probs = Dict()
    variances = Dict()
    covs = Dict()
    for element in performances
        performance = element[2]
        event = element[1]
        probs[event] = sum(performance(samples) .< 0) / sim.n
        variances[event] = (probs[event] - probs[event]^2) / sim.n
        covs[event] = sqrt(variances[event]) / probs[event]
    end
    return probs, variances, covs, samples
end

