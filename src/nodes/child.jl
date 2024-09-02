``` ContinuousChildNode
```
@auto_hash_equals struct ContinuousChildNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput}
    additional_info::Dict{Vector{Symbol},Dict}
    discretization::ApproximatedDiscretization

    function ContinuousChildNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        distribution::Dict{Vector{Symbol},<:AbstractContinuousInput},
        additional_info::Dict{Vector{Symbol},Dict},
        discretization::ApproximatedDiscretization
    )

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        !issetequal(discrete_parents, parents) && error("ContinuousChildNode $name cannot have continuous parents! Use ContinuousFunctionalNode instead")
        for key in keys(distribution)
            length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")
            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
        end

        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(distribution) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
        parents = convert(Vector{AbstractNode}, parents)
        return new(name, parents, distribution, additional_info, discretization)
    end
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput}
)
    additional_info = Dict{Vector{Symbol},Dict}()
    discretization = ApproximatedDiscretization()
    ContinuousChildNode(name, parents, distribution, additional_info, discretization)
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput},
    additional_info::Dict{Vector{Symbol},Dict}
)
    discretization = ApproximatedDiscretization()
    ContinuousChildNode(name, parents, distribution, additional_info, discretization)
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distribution::Dict{Vector{Symbol},<:AbstractContinuousInput},
    discretization::ApproximatedDiscretization
)
    additional_info = Dict{Vector{Symbol},Dict}()
    ContinuousChildNode(name, parents, distribution, additional_info, discretization)
end

function get_continuous_input(node::ContinuousChildNode, evidence::Vector{Symbol})
    node_keys = keys(node.distribution) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.distribution)]) && error("evidence $evidence does not contain all the parents of the ContinuousChildNode $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return RandomVariable(node.distribution[key], node.name)
end

function _get_node_distribution_bounds(node::ContinuousChildNode)
    lower_bound = minimum(support(i).lb for i in values(node.distribution))
    upper_bound = maximum(support(i).ub for i in values(node.distribution))
    return lower_bound, upper_bound
end

``` DiscreteChildNode
```
@auto_hash_equals struct DiscreteChildNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    states::Dict{Vector{Symbol},Dict{Symbol,AbstractDiscreteProbability}}
    additional_info::Dict{Vector{Symbol},Dict}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteChildNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        states::Dict,
        additional_info::Dict{Vector{Symbol},Dict},
        parameters::Dict{Symbol,Vector{Parameter}}
    )
        try
            states = convert(Dict{Vector{Symbol},Dict{Symbol,Real}}, states)
        catch
            try
                states = convert(Dict{Vector{Symbol},Dict{Symbol,Vector{Real}}}, states)
            catch
                error("node $name must have real valued states probailities")
            end
        end
        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        if isa(states, Dict{Vector{Symbol},Dict{Symbol,Real}})
            normalized_states = Dict{Vector{Symbol},Dict{Symbol,Real}}()
            for (key, val) in states
                verify_probabilities(val)
                normalized_prob = normalize(collect(values(val)), 1)
                normalized_states[key] = Dict(zip(collect(keys(val)), normalized_prob))
                verify_parameters(val, parameters)
                length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")
                any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
            end
            states = normalized_states
        end
        node_states = [keys(s) for s in values(states)]
        if length(reduce(intersect, node_states)) != length(reduce(union, node_states))
            error("node $name: non-coherent definition of nodes states")
        end
        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(states) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
        parents = convert(Vector{AbstractNode}, parents)

        return new(name, parents, states, additional_info, parameters)
    end
end

function DiscreteChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    states::Dict,
    additional_info::Dict{Vector{Symbol},Dict}
)
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteChildNode(name, parents, states, additional_info, parameters)
end

function DiscreteChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    states::Dict
)
    additional_info = Dict{Vector{Symbol},Dict}()
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteChildNode(name, parents, states, additional_info, parameters)
end

function DiscreteChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    states::Dict,
    parameters::Dict{Symbol,Vector{Parameter}}
)
    additional_info = Dict{Vector{Symbol},Dict}()
    DiscreteChildNode(name, parents, states, additional_info, parameters)
end

_get_states(node::DiscreteChildNode) = keys(first(values(node.states))) |> collect

function get_parameters(node::DiscreteChildNode, evidence::Vector{Symbol})
    name = node.name
    isempty(node.parameters) && error("node $name has an empty parameters vector")
    e = filter(e -> haskey(node.parameters, e), evidence)
    isempty(e) && error("evidence $evidence does not contain $name")
    return node.parameters[e[1]]
end

const global ChildNode = Union{DiscreteChildNode,ContinuousChildNode}