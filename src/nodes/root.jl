struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::D where {D<:AbstractDistribution}
end

ContinuousRootNode(rv::RandomVariable) = ContinuousRootNode(rv.name, rv.dist)

##TODO test
get_state_probability(node::ContinuousRootNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode} = node.distribution

##TODO test
function get_randomvariable(node::ContinuousRootNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    RandomVariable(node.distribution, node.name)
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

function get_state_probability(node::DiscreteRootNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    node ∉ [x[2] for x in evidence] && error("evidence does not contain DiscreteRootNode")
    [node.states[s[1]] for s in evidence if haskey(node.states, s[1])][1]
end

##TODO test
function get_parameters(node::DiscreteRootNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    node ∉ [x[2] for x in evidence] && error("evidence does not contain DiscreteRootNode")
    [node.parameters[s[1]] for s in evidence if haskey(node.parameters, s[1])][1]
end

const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}