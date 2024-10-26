@auto_hash_equals struct NewContinuousRootNode
    name::Symbol
    distribution::AbstractContinuousInput
    additional_info::Dict
    discretization::ExactDiscretization
end

NewContinuousRootNode(name::Symbol, distribution::AbstractContinuousInput, discretetization::ExactDiscretization) = NewContinuousRootNode(name, distribution, Dict(), discretetization)

NewContinuousRootNode(name::Symbol, distribution::AbstractContinuousInput) = NewContinuousRootNode(name, distribution, Dict(), ExactDiscretization())

function get_continuous_input(node::NewContinuousRootNode, ::Vector{Symbol})
    if isa(node.distribution, UnivariateDistribution)
        return RandomVariable(node.distribution, node.name)
    elseif isa(node.distribution, Tuple{Real,Real})
        return Interval(node.distribution[1], node.distribution[2], node.name)
    elseif isa(node.distribution, UnamedProbabilityBox)
        return ProbabilityBox{first(typeof(node.distribution).parameters)}(node.distribution.parameters, node.name, node.distribution.lb, node.distribution.ub)
    end
end

get_continuous_input(node::NewContinuousRootNode) = get_continuous_input(node, Vector{Symbol}())
get_continuous_input(node::NewContinuousRootNode, ::Vector{Any}) = get_continuous_input(node)

function _get_node_distribution_bounds(node::NewContinuousRootNode)
    if isa(node.distribution, UnivariateDistribution)
        lower_bound = support(node.distribution).lb
        upper_bound = support(node.distribution).ub
    elseif isa(node.distribution, Tuple{Real,Real})
        lower_bound = node.distribution[1]
        upper_bound = node.distribution[2]
    elseif isa(node.distribution, UnamedProbabilityBox)
        dist = first(typeof(node.distribution).parameters)
        if isa(dist(), Uniform)
            lower_bound = minimum(map(x -> x.lb, node.distribution.parameters))
            upper_bound = maximum(map(x -> x.ub, node.distribution.parameters))
        else
            lower_bound = support(dist).lb
            upper_bound = support(dist).ub
        end
    end
    return lower_bound, upper_bound
end

function _truncate(dist::UnivariateDistribution, i::AbstractVector)
    return truncated(dist, i[1], i[2])
end

function _truncate(dist::UnamedProbabilityBox, i::AbstractVector)
    return UnamedProbabilityBox{first(typeof(dist).parameters)}(dist.parameters, i[1], i[2])
end

function _truncate(dist::Tuple{T,T}, i::AbstractVector) where {T<:Real}
    return (i[1], i[2])
end

function _is_imprecise(node::NewContinuousRootNode)
    !isa(node.distribution, UnivariateDistribution)
end

@auto_hash_equals struct NewDiscreteRootNode
    name::Symbol
    states::Dict{Symbol,<:AbstractDiscreteProbability}
    additional_info::Dict
    parameters::Dict{Symbol,Vector{Parameter}}

    function NewDiscreteRootNode(name::Symbol, states::Dict, additional_info::Dict, parameters::Dict{Symbol,Vector{Parameter}})

        if !allequal(typeof.(values(states)))
            error("node $name has mixed interval and single value states probabilities!")
        else
            check_root_node_states!(states)
            verify_parameters(states, parameters)
            return new(name, states, additional_info, parameters)
        end
    end
end

NewDiscreteRootNode(name::Symbol, states::Dict, parameters::Dict{Symbol,Vector{Parameter}}) = NewDiscreteRootNode(name, states, Dict(), parameters)

NewDiscreteRootNode(name::Symbol, states::Dict) = NewDiscreteRootNode(name, states, Dict{Symbol,Vector{Parameter}}())

_get_states(node::NewDiscreteRootNode) = collect(keys(node.states))

function check_root_node_states!(states::Dict{Symbol,<:AbstractDiscreteProbability})
    verify_probabilities(states)
    normalize_states!(states)
end

function verify_probabilities(states::Dict{Symbol,<:Real})
    vals = collect(values(states))
    total_probability = sum(vals)
    if any(vals .< 0)
        error("probabilities must be non-negative")
    elseif any(vals .> 1)
        error("probabilities must be lower or equal than 1")
    elseif sum(vals) != 1
        if isapprox(total_probability, 1; atol=0.05)
            @warn "total probaility should be one, but the evaluated value is $total_probability , and will be normalized"
        else
            sts = collect(keys(states))
            error("states $sts are exhaustives and mutually exclusive. Their probabilities $probs does not sum up to 1")
        end
    end
end

function verify_probabilities(states::Dict{Symbol,<:AbstractVector{<:Real}})
    if any(length.(values(states)) .!= 2)
        error("interval probabilities must be defined with a 2-values vector")
    end
    probability_values = vcat(collect(values(states))...)
    if any(probability_values .< 0)
        error("probabilities must be non-negative")
    elseif any(probability_values .> 1)
        error("probabilities must be lower or equal than 1")
    elseif sum(first.(values(states))) > 1
        error("sum of intervals lower bounds is bigger than 1")
    elseif sum(last.(values(states))) < 1
        error("sum of intervals upper bounds is smaller than 1")
    end
end

function normalize_states!(states::Dict{Symbol,<:Real})
    normalized_prob = normalize(collect(values(states)), 1)
    normalized_states = Dict(zip(collect(keys(states)), normalized_prob))
    return convert(Dict{Symbol,Real}, normalized_states)
end

function normalize_states!(states::Dict{Symbol,<:AbstractVector{<:Real}})
    return states
end

function verify_parameters(states::Dict{Symbol,<:AbstractDiscreteProbability}, parameters::Dict{Symbol,Vector{Parameter}})
    if !isempty(parameters)
        if keys(states) != keys(parameters)
            error("parameters must be coherent with states")
        end
    end
end

function get_parameters(node::NewDiscreteRootNode, evidence::Vector{Symbol})
    name = node.name
    isempty(node.parameters) && error("node $name has an empty parameters vector")
    e = filter(e -> haskey(node.parameters, e), evidence)
    isempty(e) && error("evidence $evidence does not contain $name")
    return node.parameters[e[1]]
end

function _is_imprecise(node::NewDiscreteRootNode)
    any(isa.(values(node.states), Vector{Real}))
end

function _extreme_points(node::NewDiscreteRootNode)
    if _is_imprecise(node)
        new_states = _extreme_points_states_probabilities(node.states)
        return map(new_state -> NewDiscreteRootNode(node.name, new_state, node.additional_info, node.parameters), new_states)
    else
        return [node]
    end
end