struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::D where {D<:Distribution}
    intervals::Vector{Vector{Float64}}
end

ContinuousRootNode(name::Symbol, distribution::Distribution) = ContinuousRootNode(name, distribution, Vector{Vector{Float64}}())

function get_randomvariable(node::ContinuousRootNode, ::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    RandomVariable(node.distribution, node.name)
end

function is_equal(node1::ContinuousRootNode, node2::ContinuousRootNode)
    node1.name == node2.name && node1.distribution == node2.distribution && node1.intervals == node2.intervals
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

function get_parameters(node::DiscreteRootNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    all(.!is_equal.(repeat([node], length(evidence)), [x[2] for x in evidence])) && error("evidence does not contain DiscreteRootNode")
    [node.parameters[s[1]] for s in evidence if haskey(node.parameters, s[1])][1]
end

function is_equal(node1::DiscreteRootNode, node2::DiscreteRootNode)
    node1.name == node2.name && node1.states == node2.states && node1.parameters == node2.parameters
end

const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}