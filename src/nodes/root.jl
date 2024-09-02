@auto_hash_equals struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::AbstractContinuousInput
    additional_info::Dict
    discretization::ExactDiscretization # discretization just as increasing values?
end

ContinuousRootNode(name::Symbol, distribution::AbstractContinuousInput, discretetization::ExactDiscretization) = ContinuousRootNode(name, distribution, Dict(), discretetization)

ContinuousRootNode(name::Symbol, distribution::AbstractContinuousInput) = ContinuousRootNode(name, distribution, Dict(), ExactDiscretization())

function get_continuous_input(node::ContinuousRootNode, ::Vector{Symbol})
    RandomVariable(node.distribution, node.name)
end

get_continuous_input(node::ContinuousRootNode) = get_continuous_input(node, Vector{Symbol}())
get_continuous_input(node::ContinuousRootNode, ::Vector{Any}) = get_continuous_input(node)

function _get_node_distribution_bounds(node::ContinuousRootNode)
    lower_bound = support(node.distribution).lb
    upper_bound = support(node.distribution).ub
    return lower_bound, upper_bound
end

function _truncate(dist::UnivariateDistribution, i::AbstractVector)
    return truncated(dist, i[1], i[2])
end

@auto_hash_equals struct DiscreteRootNode <: DiscreteNode
    name::Symbol
    states::Dict{Symbol,<:Number}
    additional_info::Dict
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteRootNode(name::Symbol, states::Dict{Symbol,<:Number}, additional_info::Dict, parameters::Dict{Symbol,Vector{Parameter}})
        verify_probabilities(states)
        normalized_states = Dict{Symbol,Real}()
        normalized_prob = normalize(collect(values(states)), 1)
        normalized_states = Dict(zip(collect(keys(states)), normalized_prob))
        verify_parameters(normalized_states, parameters)
        return new(name, normalized_states, additional_info, parameters)
    end
end

DiscreteRootNode(name::Symbol, states::Dict, parameters::Dict{Symbol,Vector{Parameter}}) = DiscreteRootNode(name, states, Dict(), parameters)

DiscreteRootNode(name::Symbol, states::Dict) = DiscreteRootNode(name, states, Dict{Symbol,Vector{Parameter}}())

_get_states(node::DiscreteRootNode) = collect(keys(node.states))

function get_parameters(node::DiscreteRootNode, evidence::Vector{Symbol})
    name = node.name
    isempty(node.parameters) && error("node $name has an empty parameters vector")
    e = filter(e -> haskey(node.parameters, e), evidence)
    isempty(e) && error("evidence $evidence does not contain $name")
    return node.parameters[e[1]]
end

const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}