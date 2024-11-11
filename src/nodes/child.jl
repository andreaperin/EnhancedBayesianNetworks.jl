``` ContinuousChildNode
```
@auto_hash_equals struct ContinuousChildNode <: ContinuousNode
    name::Symbol
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput}
    additional_info::Dict{Vector{Symbol},Dict}
    discretization::ApproximatedDiscretization
end

function ContinuousChildNode(
    name::Symbol,
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput}
)
    additional_info = Dict{Vector{Symbol},Dict}()
    discretization = ApproximatedDiscretization()
    ContinuousChildNode(name, distribution, additional_info, discretization)
end

function ContinuousChildNode(
    name::Symbol,
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput},
    additional_info::Dict{Vector{Symbol},Dict}
)
    discretization = ApproximatedDiscretization()
    ContinuousChildNode(name, distribution, additional_info, discretization)
end

function ContinuousChildNode(
    name::Symbol,
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput},
    discretization::ApproximatedDiscretization
)
    additional_info = Dict{Vector{Symbol},Dict}()
    ContinuousChildNode(name, distribution, additional_info, discretization)
end

_get_scenarios(node::ContinuousChildNode) = collect(keys(node.distribution))

function _get_continuous_input(node::ContinuousChildNode, evidence::Vector{Symbol})
    node_keys = keys(node.distribution) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.distribution)]) && error("evidence $evidence does not contain all the parents of the ContinuousChildNode $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]

    if isa(node.distribution[key], UnivariateDistribution)
        return RandomVariable(node.distribution[key], node.name)
    elseif isa(node.distribution[key], Tuple{Real,Real})
        return Interval(node.distribution[key][1], node.distribution[key][2], node.name)
    elseif isa(node.distribution[key], UnamedProbabilityBox)
        return ProbabilityBox{first(typeof(node.distribution[key]).parameters)}(node.distribution[key].parameters, node.name, node.distribution[key].lb, node.distribution[key].ub)
    end
end

function _get_node_distribution_bounds(node::ContinuousChildNode)
    function f(x)
        if isa(x, UnivariateDistribution)
            lower_bound = support(x).lb
            upper_bound = support(x).ub
        elseif isa(x, Tuple{Real,Real})
            lower_bound = x[1]
            upper_bound = x[2]
        elseif isa(x, UnamedProbabilityBox)
            lower_bound = minimum(vcat(map(x -> x.lb, x.parameters), x.lb))
            upper_bound = maximum(vcat(map(x -> x.ub, x.parameters), x.ub))
        end
        return [lower_bound, upper_bound]
    end
    distribution_values = values(node.distribution) |> collect
    bounds = mapreduce(x -> f(x), hcat, distribution_values)
    lb = minimum(bounds[1, :])
    ub = maximum(bounds[2, :])
    return lb, ub
end

function _is_imprecise(node::ContinuousChildNode)
    any(.!isa.(values(node.distribution), UnivariateDistribution))
end

``` DiscreteChildNode
```
@auto_hash_equals struct DiscreteChildNode <: DiscreteNode
    name::Symbol
    states::Dict{Vector{Symbol},Dict{Symbol,AbstractDiscreteProbability}}
    additional_info::Dict{Vector{Symbol},Dict}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteChildNode(
        name::Symbol,
        states::Dict,
        additional_info::Dict{Vector{Symbol},Dict},
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        _check_child_states!(states)
        _verify_parameters(first(values(states)), parameters)
        return new(name, states, additional_info, parameters)
    end
end

function DiscreteChildNode(
    name::Symbol,
    states::Dict,
    additional_info::Dict{Vector{Symbol},Dict}
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteChildNode(name, states, additional_info, parameters)
end

function DiscreteChildNode(
    name::Symbol,
    states::Dict
)
    additional_info = Dict{Vector{Symbol},Dict}()
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteChildNode(name, states, additional_info, parameters)
end

function DiscreteChildNode(
    name::Symbol,
    states::Dict,
    parameters::Dict{Symbol,Vector{Parameter}}
)
    additional_info = Dict{Vector{Symbol},Dict}()
    DiscreteChildNode(name, states, additional_info, parameters)
end

_get_scenarios(node::DiscreteChildNode) = collect(keys(node.states))

_get_states(node::DiscreteChildNode) = keys(first(values(node.states))) |> collect

function _get_parameters(node::DiscreteChildNode, evidence::Vector{Symbol})
    name = node.name
    isempty(node.parameters) && error("node $name has an empty parameters vector")
    e = filter(e -> haskey(node.parameters, e), evidence)
    isempty(e) && error("evidence $evidence does not contain $name")
    return node.parameters[e[1]]
end

function _is_imprecise(node::DiscreteChildNode)
    probability_values = values(node.states) |> collect
    probability_values = vcat(collect.(values.(probability_values))...)
    any(isa.(probability_values, AbstractVector{<:Real}))
end

function _extreme_points(node::DiscreteChildNode)
    if _is_imprecise(node)
        new_states = map(states -> _extreme_points_states_probabilities(states), values(node.states))
        new_states_combination = vec(collect(Iterators.product(new_states...)))
        new_states = map(nsc -> Dict(keys(node.states) .=> nsc), new_states_combination)
        return map(new_state -> DiscreteChildNode(node.name, new_state, node.additional_info, node.parameters), new_states)
    else
        return [node]
    end
end

const global ChildNode = Union{DiscreteChildNode,ContinuousChildNode}