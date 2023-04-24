struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::D where {D<:Distribution}
end

ContinuousRootNode(rv::RandomVariable) = ContinuousRootNode(rv.name, rv.dist)
struct DiscreteRootNode <: DiscreteNode
    name::Symbol
    states::Dict{Symbol,<:Real}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Real}, parameters::Dict{Symbol,Vector{Parameter}})
        verify_probabilities(states)
        return new(name, states, parameters)
    end
end

DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Real}) = DiscreteRootNode(name, states, Dict{Symbol,Vector{Parameter}}())

_get_states(node::DiscreteRootNode) = collect(keys(node.states))


const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}