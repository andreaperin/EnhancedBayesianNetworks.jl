``` NewContinuousChildNode
```
@auto_hash_equals struct NewContinuousChildNode
    name::Symbol
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput}
    additional_info::Dict{Vector{Symbol},Dict}
    discretization::ApproximatedDiscretization
end

function NewContinuousChildNode(
    name::Symbol,
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput}
)
    additional_info = Dict{Vector{Symbol},Dict}()
    discretization = ApproximatedDiscretization()
    NewContinuousChildNode(name, distribution, additional_info, discretization)
end

function NewContinuousChildNode(
    name::Symbol,
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput},
    additional_info::Dict{Vector{Symbol},Dict}
)
    discretization = ApproximatedDiscretization()
    NewContinuousChildNode(name, distribution, additional_info, discretization)
end

function NewContinuousChildNode(
    name::Symbol,
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput},
    discretization::ApproximatedDiscretization
)
    additional_info = Dict{Vector{Symbol},Dict}()
    NewContinuousChildNode(name, distribution, additional_info, discretization)
end

function get_continuous_input(node::NewContinuousChildNode, evidence::Vector{Symbol})
    node_keys = keys(node.distribution) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.distribution)]) && error("evidence $evidence does not contain all the parents of the NewContinuousChildNode $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]

    if isa(node.distribution[key], UnivariateDistribution)
        return RandomVariable(node.distribution[key], node.name)
    elseif isa(node.distribution[key], Tuple{Real,Real})
        return Interval(node.distribution[key][1], node.distribution[key][2], node.name)
    elseif isa(node.distribution[key], UnamedProbabilityBox)
        return ProbabilityBox{first(typeof(node.distribution[key]).parameters)}(node.distribution[key].parameters, node.name, node.distribution[key].lb, node.distribution[key].ub)
    end
end

function _get_node_distribution_bounds(node::NewContinuousChildNode)
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

function _is_imprecise(node::NewContinuousChildNode)
    any(.!isa.(values(node.distribution), UnivariateDistribution))
end

``` NewDiscreteChildNode
```
@auto_hash_equals struct NewDiscreteChildNode
    name::Symbol
    states::Dict{Vector{Symbol},Dict{Symbol,AbstractDiscreteProbability}}
    additional_info::Dict{Vector{Symbol},Dict}
    parameters::Dict{Symbol,Vector{Parameter}}

    function NewDiscreteChildNode(
        name::Symbol,
        states::Dict,
        additional_info::Dict{Vector{Symbol},Dict},
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        new_states = Dict()
        for (key, val) in states
            if !allequal(typeof.(values(val)))
                error("node $name has mixed interval and single value states probabilities!")
            else
                new_states[key] = _verify_child_node_state!(val, parameters)
            end
        end
        return new(name, new_states, additional_info, parameters)
    end
end

function NewDiscreteChildNode(
    name::Symbol,
    states::Dict,
    additional_info::Dict{Vector{Symbol},Dict}
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    NewDiscreteChildNode(name, states, additional_info, parameters)
end

function NewDiscreteChildNode(
    name::Symbol,
    states::Dict
)
    additional_info = Dict{Vector{Symbol},Dict}()
    parameters = Dict{Symbol,Vector{Parameter}}()
    NewDiscreteChildNode(name, states, additional_info, parameters)
end

function NewDiscreteChildNode(
    name::Symbol,
    states::Dict,
    parameters::Dict{Symbol,Vector{Parameter}}
)

    additional_info = Dict{Vector{Symbol},Dict}()
    NewDiscreteChildNode(name, states, additional_info, parameters)
end

function _verify_discrete_child_node_state(states::Dict)
    node_states = [keys(s) for s in values(states)]
    if length(reduce(intersect, node_states)) != length(reduce(union, node_states))
        state_list = unique(collect(Iterators.Flatten(node_states)))
        error("non-coherent definition of nodes states: $state_list")
    end
end

_get_states(node::NewDiscreteChildNode) = keys(first(values(node.states))) |> collect

function get_parameters(node::NewDiscreteChildNode, evidence::Vector{Symbol})
    name = node.name
    isempty(node.parameters) && error("node $name has an empty parameters vector")
    e = filter(e -> haskey(node.parameters, e), evidence)
    isempty(e) && error("evidence $evidence does not contain $name")
    return node.parameters[e[1]]
end

function _is_imprecise(node::NewDiscreteChildNode)
    probability_values = values(node.states) |> collect
    probability_values = vcat(collect.(values.(probability_values))...)
    any(isa.(probability_values, Vector{Real}))
end

function _extreme_points(node::NewDiscreteChildNode)
    if _is_imprecise(node)
        new_states = map(states -> _extreme_points_states_probabilities(states), values(node.states))
        new_states_combination = vec(collect(Iterators.product(new_states...)))

        new_states = map(nsc -> Dict(keys(node.states) .=> nsc), new_states_combination)
        return map(new_state -> NewDiscreteChildNode(node.name, new_state, node.additional_info, node.parameters), new_states)
    else
        return [node]
    end
end