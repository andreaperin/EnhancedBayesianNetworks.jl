function verify_probabilities(states::Dict{Symbol,<:Real})
    any(values(states) .< 0.0) && error("Probabilites must be nonnegative")
    any(values(states) .> 1.0) && error("Probabilites must be less or equal to 1.0")
    sum(values(states)) != 1 && error("Probabilites must sum up to 1.0")
end

function verify_parameters(states::Dict{Symbol,<:Real}, parameters::Dict{Symbol,Vector{Parameter}})
    if !isempty(parameters)
        keys(states) != keys(parameters) && error("parameters must be coherent with states")
    end
end

function verify_functionalnode_parents(parents::Vector{<:AbstractNode})
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    discrete_parents_states = vcat([collect(keys(x.states)) for x in discrete_parents]...)

    any(isempty.([x.parameters for x in discrete_parents])) && error("all discrete parents of a functional node must have a non-empty parameters dictionary")

    unique(discrete_parents_states) != discrete_parents_states && error("all discrete parents of a functional node must have different named states")
end
