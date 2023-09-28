function verify_probabilities(states::Dict{Symbol,<:Real})
    any(values(states) .< 0.0) && error("Probabilites must be nonnegative")
    any(values(states) .> 1.0) && error("Probabilites must be less or equal to 1.0")
    total_proability = sum(values(states))
    if !isapprox(total_proability, 1)
        @warn "total probaility should be one, but the evaluated value is $total_proability , and will be normalized"
    end
end

function verify_parameters(states::Dict{Symbol,<:Real}, parameters::Dict{Symbol,Vector{Parameter}})
    if !isempty(parameters)
        keys(states) != keys(parameters) && error("parameters must be coherent with states")
    end
end

function verify_functionalnode_parents(parents::Vector{<:AbstractNode})
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    discrete_parents_states = vcat([collect(keys(x.states)) for x in discrete_parents]...)
    discrete_parents_states = mapreduce(x -> EnhancedBayesianNetworks._get_states(x), vcat, discrete_parents)

    any(isempty.([x.parameters for x in discrete_parents])) && error("all discrete parents of a functional node must have a non-empty parameters dictionary")

    unique(discrete_parents_states) != discrete_parents_states && error("all discrete parents of a functional node must have different named states")
end

function Base.isequal(node1::AbstractNode, node2::AbstractNode)
    typeof(node1) == typeof(node2) && isequal(node1, node2)
end
