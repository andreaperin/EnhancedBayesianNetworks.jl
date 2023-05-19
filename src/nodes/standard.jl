mutable struct ContinuousStandardNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    distribution::OrderedDict{Vector{Symbol},D} where {D<:Distribution}
    intervals::Vector{Vector{Float64}}
    sigma::Real

    function ContinuousStandardNode(name::Symbol,
        parents::Vector{<:AbstractNode},
        distribution::OrderedDict{Vector{Symbol},D},
        intervals::Vector{Vector{Float64}},
        sigma::Real
    ) where {D<:Distribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        Set(discrete_parents) != Set(parents) && error("ContinuousStandardNode cannot have continuous parents, use ContinuousFunctionalNode instead")
        for (key, _) in distribution
            length(discrete_parents) != length(key) && error("Number of symbols per parent in node.states must be equal to the number of discrete parents")

            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(distribution) && error("defined combinations in node.states must be equal to the theorical discrete parents combinations")
        return new(name, parents, distribution, intervals, sigma)
    end
end

ContinuousStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, distribution::OrderedDict{Vector{Symbol},D}) where {D<:Distribution} = ContinuousStandardNode(name, parents, distribution, Vector{Vector{Float64}}(), 0)

function get_state_probability(node::ContinuousStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    all([i.name for i in node.parents] .∉ [[x[2].name for x in evidence]]) && error("evidence does not contain any parents of the ContinuousStandardNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return node.distribution[node_key]
end

function get_randomvariable(node::ContinuousStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    all(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the ContinuousStandardNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end

    RandomVariable(node.distribution[node_key], node.name)
end

mutable struct DiscreteStandardNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    states::OrderedDict{Vector{Symbol},Dict{Symbol,T}} where {T<:Real}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}, parameters::Dict{Symbol,Vector{Parameter}}) where {T<:Real}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, val) in states
            verify_probabilities(val)
            verify_parameters(val, parameters)
            length(discrete_parents) != length(key) && error("number of symbols per parent in node.states must be equal to the number of discrete parents")
            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents")
        end

        node_states = [keys(s) for s in values(states)]
        if length(reduce(intersect, node_states)) != length(reduce(union, node_states))
            error("NON coherent definition of nodes states in the ordered dict")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(states) && error("defined combinations in node.states must be equal to the theorical discrete parents combinations")

        return new(name, parents, states, parameters)
    end
end

function DiscreteStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}) where {T<:Real}
    DiscreteStandardNode(name, parents, states, Dict{Symbol,Vector{Parameter}}())
end

_get_states(node::DiscreteStandardNode) = keys(first(values(node.states))) |> collect

function get_state_probability(node::DiscreteStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    all(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the DiscreteStandardNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return node.states[node_key]
end

function get_parameters(node::DiscreteStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    node ∉ [x[2] for x in evidence] && error("evidence does not contain DiscreteStandardNode in the evidence")
    node_key = [e[1] for e in evidence if e[2] == node][1]
    return node.parameters[node_key]
end



const global StandardNode = Union{DiscreteStandardNode,ContinuousStandardNode}