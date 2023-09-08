struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::Distribution
    intervals::Vector{Vector{Float64}}
end

ContinuousRootNode(name::Symbol, distribution::Distribution) = ContinuousRootNode(name, distribution, Vector{Vector{Float64}}())

function get_randomvariable(node::ContinuousRootNode, ::Vector{Symbol})
    RandomVariable(node.distribution, node.name)
end

function Base.isequal(node1::ContinuousRootNode, node2::ContinuousRootNode)
    node1.name == node2.name && node1.distribution == node2.distribution && node1.intervals == node2.intervals
end

function Base.hash(node::ContinuousRootNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.distribution, h)
    h = hash(node.intervals, h)
    return h
end

struct DiscreteRootNode <: DiscreteNode
    name::Symbol
    states::Dict{Symbol,<:Real}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Real}, parameters::Dict{Symbol,Vector{Parameter}})
        verify_probabilities(states)
        verify_parameters(states, parameters)
        return new(name, states, parameters)
    end
end

DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Real}) = DiscreteRootNode(name, states, Dict{Symbol,Vector{Parameter}}())

_get_states(node::DiscreteRootNode) = collect(keys(node.states))

function get_parameters(node::DiscreteRootNode, evidence::Vector{Symbol})
    name = node.name
    isempty(node.parameters) && error("node $name has an empty parameters vector")
    e = filter(e -> haskey(node.parameters, e), evidence)
    isempty(e) && error("evidence $evidence does not contain $name")
    return node.parameters[e[1]]
end

function Base.isequal(node1::DiscreteRootNode, node2::DiscreteRootNode)
    node1.name == node2.name && node1.states == node2.states && node1.parameters == node2.parameters
end

function Base.hash(node::DiscreteRootNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.states, h)
    h = hash(node.parameters, h)
    return h
end

const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}