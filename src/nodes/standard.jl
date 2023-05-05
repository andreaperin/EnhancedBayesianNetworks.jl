## TODO Add check for each standard node for order of the states keys coherent with parents order!

struct ContinuousStandardNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    distribution::OrderedDict{Vector{Symbol},D} where {D<:Distribution}

    function ContinuousStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, distribution::OrderedDict{Vector{Symbol},D}) where {D<:Distribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)

        for (key, _) in distribution
            length(discrete_parents) != length(key) && error("number of symbols per parent in node.states must be equal to the number of discrete parents")
            ##TODO test
            any([key[i] ∉ keys(discrete_parents[i].states) for i in range(1, length(key))]) && error("order of discrete parents must be equal to the order of states")
        end

        discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
        discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
        length(discrete_parents_combination) != length(distribution) && error("defined combinations in node.states must be equal to the theorical discrete parents combinations")
        any(discrete_parents_combination .∉ [keys(distribution)]) && error("missmatch in defined parents combinations states and states of the parents")

        return new(name, parents, distribution)
    end
end

##TODO test
function get_state_probability(node::ContinuousStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    any(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the ContinuousStandardNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return node.distribution[node_key]
end

##TODO test
function get_randomvariable(node::ContinuousStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    any(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the ContinuousStandardNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return RandomVariable(node.distribution[node_key], node.name)
end

struct DiscreteStandardNode <: DiscreteNode
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
            ##TODO test
            any([key[i] ∉ keys(discrete_parents[i].states) for i in range(1, length(key))]) && error("order of discrete parents must be equal to the order of states")
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

_get_states(node::DiscreteStandardNode) = keys(first(values(node.states))) |> collect

##TODO test
function get_state_probability(node::DiscreteStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    any(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the DiscreteStandardNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return node.states[node_key]
end

##TODO test
function get_parameters(node::DiscreteStandardNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    node ∉ [x[2] for x in evidence] && error("evidence does not contain DiscreteStandardNode in the evidence")
    node_key = [e[1] for e in evidence if e[2] == node][1]
    return node.parameters[node_key]
end



const global StandardNode = Union{DiscreteStandardNode,ContinuousStandardNode}
