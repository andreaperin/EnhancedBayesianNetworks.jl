struct ContinuousStandardNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    distribution::OrderedDict{Vector{Symbol},D} where {D<:Distribution}

    function ContinuousStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, distribution::OrderedDict{Vector{Symbol},D}) where {D<:Distribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, _) in distribution
            length(discrete_parents) != length(key) && error("number of symbols per parent in node.states must be equal to the number of discrete parents")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(distribution) && error("defined combinations in node.states must be equal to the theorical discrete parents combinations")
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
    parents::Vector{<:AbstractNode}
    states::OrderedDict{Vector{Symbol},Dict{Symbol,T}} where {T<:Real}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}, parameters::Dict{Symbol,Vector{Parameter}}) where {T<:Real}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, val) in states
            _not_negative(val) && error("Probabilites must be nonnegative")
            _less_than_one(val) && error("Probabilites must be less or equal to 1.0")
            _sum_up_to_one(val) && error("Probabilites must sum up to 1.0")

            length(discrete_parents) != length(key) && error("number of symbols per parent in node.states must be equal to the number of discrete parents")
        end

        node_states = [keys(s) for s in values(states)]
        if length(reduce(intersect, node_states)) != length(reduce(union, node_states))
            error("NON coherent definition of nodes states in the ordered dict")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(states) && error("defined combinations in node.states must be equal to the theorical discrete parents combinations")
        any(discrete_parents_combination .∉ [keys(states)]) && error("missmatch in defined parents combinations states and states of the parents")

        return new(name, parents, states, parameters)
    end
end

function DiscreteStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}) where {T<:Real}
    DiscreteStandardNode(name, parents, states, Dict{Symbol,Vector{Parameter}}())
end

function _get_states(node::DiscreteStandardNode)
    return keys(first(values(node.states))) |> collect
end

const global StandardNode = Union{DiscreteStandardNode,ContinuousStandardNode}
