struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::Distribution
    discretization::ExactDiscretization # discretization just as increasing values?
end

ContinuousRootNode(name::Symbol, distribution::Distribution) = ContinuousRootNode(name, distribution, ExactDiscretization())

function get_randomvariable(node::ContinuousRootNode, ::Vector{Symbol})
    RandomVariable(node.distribution, node.name)
end

get_randomvariable(node::ContinuousRootNode) = get_randomvariable(node, Vector{Symbol}())

function Base.isequal(node1::ContinuousRootNode, node2::ContinuousRootNode)
    node1.name == node2.name && node1.distribution == node2.distribution && isequal(node1.discretization, node2.discretization)
end

function Base.hash(node::ContinuousRootNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.distribution, h)
    h = hash(node.discretization, h)
    return h
end

struct DiscreteRootNode <: DiscreteNode
    name::Symbol
    states::Dict{Symbol,<:Number}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Number}, parameters::Dict{Symbol,Vector{Parameter}})
        verify_probabilities(states)
        normalized_states = Dict{Symbol,Number}()
        normalized_prob = normalize(collect(values(states)), 1)
        normalized_states = Dict(zip(collect(keys(states)), normalized_prob))
        verify_parameters(normalized_states, parameters)
        return new(name, normalized_states, parameters)
    end
end

DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Number}) = DiscreteRootNode(name, states, Dict{Symbol,Vector{Parameter}}())

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