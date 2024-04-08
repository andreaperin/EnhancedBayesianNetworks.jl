@auto_hash_equals struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::AbstractContinuousInput
    discretization::ExactDiscretization # discretization just as increasing values?
end

ContinuousRootNode(name::Symbol, distribution::AbstractContinuousInput) = ContinuousRootNode(name, distribution, ExactDiscretization())

function get_randomvariable(node::ContinuousRootNode, ::Vector{Symbol})
    if isa(node.distribution, UnivariateDistribution)
        return RandomVariable(node.distribution, node.name)
    elseif isa(node.distribution, Tuple{Real,Real})
        return Interval(node.distribution[1], node.distribution[2], node.name)
    end
end

get_randomvariable(node::ContinuousRootNode) = get_randomvariable(node, Vector{Symbol}())

function _get_node_distribution_bounds(node::ContinuousRootNode)
    if isa(node.distribution, UnivariateDistribution)
        lower_bound = support(node.distribution).lb
        upper_bound = support(node.distribution).ub
    elseif isa(node.distribution, Tuple{Real,Real})
        lower_bound = node.distribution[1]
        upper_bound = node.distribution[2]
    end
    return lower_bound, upper_bound
end

function _truncate(dist::UnivariateDistribution, i::AbstractVector)
    return truncated(dist, i[1], i[2])
end

function _is_imprecise(node::ContinuousRootNode)
    !isa(node.distribution, UnivariateDistribution)
end

@auto_hash_equals struct DiscreteRootNode <: DiscreteNode
    name::Symbol
    states::Dict{Symbol,AbstractDiscreteProbability}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteRootNode(name::Symbol, states::Dict, parameters::Dict{Symbol,Vector{Parameter}})
        if all(isa.(values(states), Real))
            verify_probabilities(states)
            normalized_states = Dict{Symbol,Real}()
            normalized_prob = normalize(collect(values(states)), 1)
            normalized_states = Dict(zip(collect(keys(states)), normalized_prob))
            verify_parameters(normalized_states, parameters)
            return new(name, normalized_states, parameters)

        else
            if any(isa.(values(states), Real))
                error("node $name has mixed interval and single value states probabilities!")
            else
                try
                    states = convert(Dict{Symbol,AbstractVector{Real}}, states)
                catch
                    error("node $name must have real valued states probabilities")
                end
                verify_interval_probabilities(states)
                verify_parameters(states, parameters)
                return new(name, states, parameters)
            end
        end
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

function _is_imprecise(node::DiscreteRootNode)
    any(isa.(values(node.states), Vector{Real}))
end

const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}