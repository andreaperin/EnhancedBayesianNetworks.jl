function verify_probabilities(states::Dict{Symbol,<:Real})
    any(values(states) .< 0.0) && error("Probabilites must be nonnegative")
    any(values(states) .> 1.0) && error("Probabilites must be less or equal to 1.0")
    total_proability = sum(values(states))
    if total_proability != 1
        if isapprox(total_proability, 1; rtol=0.01)
            @warn "total probaility should be one, but the evaluated value is $total_proability , and will be normalized"
        else
            error("defined states probabilities $states are wrong")
        end
    end
end

function verify_probabilities(states::Dict{Symbol,Vector{<:Real}})
    probabilities_values = (values.(values(states))) .|> collect
    flatten_probabilities_values = collect(Iterators.flatten(Iterators.flatten(probabilities_values)))
    any(flatten_probabilities_values .< 0.0) && error("Probabilites must be nonnegative")
    any(flatten_probabilities_values .> 1.0) && error("Probabilites must be less or equal to 1.0")
end

function verify_parameters(states::Dict, parameters::Dict{Symbol,Vector{Parameter}})
    if !isempty(parameters)
        keys(states) != keys(parameters) && error("parameters must be coherent with states")
    end
end

function verify_functionalnode_parents(parents::Vector{<:AbstractNode})
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    discrete_parents = filter(x -> !isa(x, FunctionalNode), discrete_parents)
    if !isempty(discrete_parents)
        discrete_parents_states = vcat([collect(keys(x.states)) for x in discrete_parents]...)
        discrete_parents_states = mapreduce(x -> _get_states(x), vcat, discrete_parents)

        any(isempty.([x.parameters for x in discrete_parents])) && error("all discrete parents of a functional node must have a non-empty parameters dictionary")

        unique(discrete_parents_states) != discrete_parents_states && error("all discrete parents of a functional node must have different named states")
    end
end

