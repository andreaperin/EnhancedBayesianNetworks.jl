struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::D where {D<:Distribution}
end

ContinuousRootNode(rv::RandomVariable) = ContinuousRootNode(rv.name, rv.dist)

_get_states(node::ContinuousNode) = node.distribution
struct DiscreteRootNode <: DiscreteNode
    name::Symbol
    states::Dict{Symbol,<:Real}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Real}, parameters::Dict{Symbol,Vector{Parameter}})

        _not_negative(states) && error("Probabilites must be nonnegative")
        _less_than_one(states) && error("Probabilites must be less or equal to 1.0")
        _sum_up_to_one(states) && error("Probabilites must sum up to 1.0")

        return new(name, states, parameters)
    end
end

DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Real}) = DiscreteRootNode(name, states, Dict{Symbol,Vector{Parameter}}())

_get_states(node::DiscreteRootNode) = collect(keys(node.states))


const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}