struct ContinuousStandardNode <: ContinuousNode
    name::Symbol
    parents::Vector{N} where {N<:AbstractNode}
    distribution::OrderedDict{Vector{Symbol},D} where {D<:Distribution}

    function ContinuousStandardNode(name::Symbol, parents::Vector{N}, distribution::OrderedDict{Vector{Symbol},D}) where {N<:AbstractNode,D<:Distribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, val) in distribution
            length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(distribution) && error("defined combinations must be equal to the discrete parents combinations")
        any(discrete_parents_combination .∉ [keys(distribution)]) && error("missmatch in defined parents combinations states and states of the parents")

        return new(name, parents, distribution)
    end
end

function _get_states(node::ContinuousStandardNode)
    list = collect(values(node.distribution))
    return list
end

struct DiscreteStandardNode <: DiscreteNode
    name::Symbol
    parents::Vector{N} where {N<:AbstractNode}
    states::OrderedDict{Vector{Symbol},Dict{Symbol,T}} where {T<:Real}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteStandardNode(name::Symbol, parents::Vector{N}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}, parameters::Dict{Symbol,Vector{Parameter}}) where {N<:AbstractNode,T<:Real}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, val) in states
            any(values(val) .< 0.0) && error("Probabilites must be nonnegative")
            any(values(val) .> 1.0) && error("Probabilites must be less or equal to 1.0")
            sum(values(val)) > 1.0 && error("Probabilites must sum up to 1.0")

            length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")
        end

        node_states = [keys(s) for s in values(states)]
        if length(reduce(intersect, node_states)) != length(reduce(union, node_states))
            error("non coherent definition of nodes states in the ordered dict")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(states) && error("defined combinations must be equal to the discrete parents combinations")
        any(discrete_parents_combination .∉ [keys(states)]) && error("missmatch in defined parents combinations states and states of the parents")

        return new(name, parents, states, parameters)
    end
end

function DiscreteStandardNode(name::Symbol, parents::Vector{N}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}) where {N<:AbstractNode,T<:Real}
    DiscreteStandardNode(name, parents, states, Dict{Symbol,Vector{Parameter}}())
end

function _get_states(node::DiscreteStandardNode)
    return keys(first(values(node.states))) |> collect
end

const global StandardNode = Union{DiscreteStandardNode,ContinuousStandardNode}
