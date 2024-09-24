@auto_hash_equals struct ContinuousRootNode <: ContinuousNode
    name::Symbol
    distribution::AbstractContinuousInput
    additional_info::Dict
    discretization::ExactDiscretization # discretization just as increasing values?
end

ContinuousRootNode(name::Symbol, distribution::AbstractContinuousInput, discretetization::ExactDiscretization) = ContinuousRootNode(name, distribution, Dict(), discretetization)

ContinuousRootNode(name::Symbol, distribution::AbstractContinuousInput) = ContinuousRootNode(name, distribution, Dict(), ExactDiscretization())

function get_continuous_input(node::ContinuousRootNode, ::Vector{Symbol})
    if isa(node.distribution, UnivariateDistribution)
        return RandomVariable(node.distribution, node.name)
    elseif isa(node.distribution, Tuple{Real,Real})
        return Interval(node.distribution[1], node.distribution[2], node.name)
    elseif isa(node.distribution, AbstractProbabilityBox)
        return ProbabilityBox{first(typeof(node.distribution).parameters)}(node.distribution.parameters, node.name)
    end
end

get_continuous_input(node::ContinuousRootNode) = get_continuous_input(node, Vector{Symbol}())
get_continuous_input(node::ContinuousRootNode, ::Vector{Any}) = get_continuous_input(node)

function _get_node_distribution_bounds(node::ContinuousRootNode)
    if isa(node.distribution, UnivariateDistribution)
        lower_bound = support(node.distribution).lb
        upper_bound = support(node.distribution).ub
    elseif isa(node.distribution, Tuple{Real,Real})
        lower_bound = node.distribution[1]
        upper_bound = node.distribution[2]
    elseif isa(node.distribution, AbstractProbabilityBox)
        p_box = ProbabilityBox{first(typeof(node.distribution).parameters)}(node.distribution.parameters, node.name)
        lower_bound = p_box.lb
        upper_bound = p_box.ub
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
    additional_info::Dict
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteRootNode(name::Symbol, states::Dict, additional_info::Dict, parameters::Dict{Symbol,Vector{Parameter}})
        if all(isa.(values(states), Real))
            verify_probabilities(states)
            normalized_states = Dict{Symbol,Real}()
            normalized_prob = normalize(collect(values(states)), 1)
            normalized_states = Dict(zip(collect(keys(states)), normalized_prob))
            verify_parameters(normalized_states, parameters)
            return new(name, normalized_states, additional_info, parameters)

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
                return new(name, states, additional_info, parameters)
            end
        end
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

function _is_imprecise(node::DiscreteRootNode)
    any(isa.(values(node.states), Vector{Real}))
end

function _extreme_points(node::DiscreteRootNode)
    if EnhancedBayesianNetworks._is_imprecise(node)
        new_states = _extreme_points_states_probabilities(node.states)
        return map(new_state -> DiscreteRootNode(node.name, new_state, node.additional_info, node.parameters), new_states)
    else
        return [node]
    end
end

const global RootNode = Union{DiscreteRootNode,ContinuousRootNode}