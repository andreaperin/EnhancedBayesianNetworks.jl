mutable struct ContinuousStandardNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    distribution::OrderedDict{Vector{Symbol},Distribution}
    intervals::Vector{Vector{Float64}}
    sigma::Real

    function ContinuousStandardNode(name::Symbol,
        parents::Vector{<:AbstractNode},
        distribution::OrderedDict{Vector{Symbol},D},
        intervals::Vector{Vector{Float64}},
        sigma::Real
    ) where {D<:Distribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        !issetequal(discrete_parents, parents) && error("ContinuousStandardNode cannot have continuous parents, use ContinuousFunctionalNode instead")
        for key in keys(distribution)
            length(discrete_parents) != length(key) && error("Number of symbols per parent in node.states must be equal to the number of discrete parents")

            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents")
        end

        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(distribution) && error("defined combinations in node.states must be equal to the theorical discrete parents combinations")
        return new(name, parents, distribution, intervals, sigma)
    end
end

ContinuousStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, distribution::OrderedDict{Vector{Symbol},D}) where {D<:Distribution} = ContinuousStandardNode(name, parents, distribution, Vector{Vector{Float64}}(), 0)


function get_randomvariable(node::ContinuousStandardNode, evidence::Vector{Symbol})
    node_keys = keys(node.distribution) |> collect
    all(.![issubset(i, evidence) for i in keys(node.distribution)]) && error("evidence does not contain all the parents of the ContinuousStandardNode")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return RandomVariable(node.distribution[key], node.name)
end

function Base.isequal(node1::ContinuousStandardNode, node2::ContinuousStandardNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.distributions == node2.distributions && node1.intervals == node2.intervals && node1.sigma == node2.sigma
end

function Base.hash(node::ContinuousStandardNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.distributions, h)
    h = hash(node.intervals, h)
    h = hash(node.sigma, h)

    return h
end

mutable struct DiscreteStandardNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    states::OrderedDict{Vector{Symbol},Dict{Symbol,Real}}
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

        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(states) && error("defined combinations in node.states must be equal to the theorical discrete parents combinations")

        return new(name, parents, states, parameters)
    end
end

function DiscreteStandardNode(name::Symbol, parents::Vector{<:AbstractNode}, states::OrderedDict{Vector{Symbol},Dict{Symbol,T}}) where {T<:Real}
    DiscreteStandardNode(name, parents, states, Dict{Symbol,Vector{Parameter}}())
end

_get_states(node::DiscreteStandardNode) = keys(first(values(node.states))) |> collect

function get_parameters(node::DiscreteStandardNode, evidence::Vector{Symbol})
    isempty(node.parameters) && error("node has an empty parameters vector")
    e = filter(e -> haskey(node.parameters, e), evidence)
    isempty(e) && error("evidence does not contain DiscreteStandardNode")
    return node.parameters[e[1]]
end


function Base.isequal(node1::DiscreteStandardNode, node2::DiscreteStandardNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.states == node2.states && node1.parameters == node2.parameters
end

function Base.hash(node::DiscreteStandardNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.states, h)
    h = hash(node.parameters, h)

    return h
end


const global StandardNode = Union{DiscreteStandardNode,ContinuousStandardNode}