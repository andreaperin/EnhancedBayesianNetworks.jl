``` ContinuousChildNode
```
@auto_hash_equals struct ContinuousChildNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    distributions::Dict{Vector{Symbol},UnivariateDistribution}
    samples::Dict{Vector{Symbol},DataFrame}
    discretization::ApproximatedDiscretization

    function ContinuousChildNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        distributions::Dict{Vector{Symbol},D},
        samples::Dict{Vector{Symbol},DataFrame},
        discretization::ApproximatedDiscretization
    ) where {D<:UnivariateDistribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        !issetequal(discrete_parents, parents) && error("ContinuousChildNode $name cannot have continuous parents! Use ContinuousFunctionalNode instead")
        for key in keys(distributions)
            length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")
            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
        end

        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(distributions) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
        parents = convert(Vector{AbstractNode}, parents)
        return new(name, parents, distributions, samples, discretization)
    end
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distributions::Dict{Vector{Symbol},D}
) where {D<:UnivariateDistribution}

    samples = Dict{Vector{Symbol},DataFrame}()
    discretization = ApproximatedDiscretization()
    ContinuousChildNode(name, parents, distributions, samples, discretization)
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distributions::Dict{Vector{Symbol},D},
    samples::Dict{Vector{Symbol},DataFrame}
) where {D<:UnivariateDistribution}

    discretization = ApproximatedDiscretization()
    ContinuousChildNode(name, parents, distributions, samples, discretization)
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distributions::Dict{Vector{Symbol},D},
    discretization::ApproximatedDiscretization
) where {D<:UnivariateDistribution}

    samples = Dict{Vector{Symbol},DataFrame}()
    ContinuousChildNode(name, parents, distributions, samples, discretization)
end

function get_randomvariable(node::ContinuousChildNode, evidence::Vector{Symbol})
    node_keys = keys(node.distributions) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.distributions)]) && error("evidence $evidence does not contain all the parents of the ContinuousChildNode $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return RandomVariable(node.distributions[key], node.name)
end

function _get_node_distribution_bounds(node::ContinuousChildNode)
    lower_bound = minimum(support(i).lb for i in values(node.distributions))
    upper_bound = maximum(support(i).ub for i in values(node.distributions))
    return lower_bound, upper_bound
end

``` DiscreteChildNode
```
@auto_hash_equals struct DiscreteChildNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    states::Dict{Vector{Symbol},Dict{Symbol,Real}}
    covs::Dict{Vector{Symbol},Number}
    samples::Dict{Vector{Symbol},DataFrame}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteChildNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        states::Dict{Vector{Symbol},Dict{Symbol,T}},
        covs::Dict{Vector{Symbol},Number},
        samples::Dict{Vector{Symbol},DataFrame},
        parameters::Dict{Symbol,Vector{Parameter}}
    ) where {T<:Real}

        normalized_states = Dict{Vector{Symbol},Dict{Symbol,Real}}()
        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        for (key, val) in states
            verify_probabilities(val)
            normalized_prob = normalize(collect(values(val)), 1)
            normalized_states[key] = Dict(zip(collect(keys(val)), normalized_prob))
            verify_parameters(val, parameters)
            length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")
            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
        end

        node_states = [keys(s) for s in values(normalized_states)]
        if length(reduce(intersect, node_states)) != length(reduce(union, node_states))
            error("node $name: non-coherent definition of nodes states")
        end

        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(normalized_states) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
        parents = convert(Vector{AbstractNode}, parents)

        return new(name, parents, normalized_states, covs, samples, parameters)
    end
end

function DiscreteChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    states::Dict{Vector{Symbol},Dict{Symbol,T}},
    covs::Dict{Vector{Symbol},Number},
    samples::Dict{Vector{Symbol},DataFrame}
) where {T<:Real}

    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteChildNode(name, parents, states, covs, samples, parameters)
end

function DiscreteChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    states::Dict{Vector{Symbol},Dict{Symbol,T}}
) where {T<:Real}

    covs = Dict{Vector{Symbol},Number}()
    samples = Dict{Vector{Symbol},DataFrame}()
    parameters = Dict{Symbol,Vector{Parameter}}()
    DiscreteChildNode(name, parents, states, covs, samples, parameters)
end

function DiscreteChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    states::Dict{Vector{Symbol},Dict{Symbol,T}},
    parameters::Dict{Symbol,Vector{Parameter}}
) where {T<:Real}

    covs = Dict{Vector{Symbol},Number}()
    samples = Dict{Vector{Symbol},DataFrame}()
    DiscreteChildNode(name, parents, states, covs, samples, parameters)
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