``` ContinuousChildNode
```

mutable struct ContinuousChildNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    distributions::Dict{Vector{Symbol},Distribution}
    samples::Dict{Vector{Symbol},DataFrame}
    discretization::ApproximatedDiscretization

    function ContinuousChildNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        distributions::Dict{Vector{Symbol},D},
        samples::Dict{Vector{Symbol},DataFrame},
        discretization::ApproximatedDiscretization
    ) where {D<:Distribution}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        !issetequal(discrete_parents, parents) && error("ContinuousChildNode $name cannot have continuous parents! Use ContinuousFunctionalNode instead")
        for key in keys(distributions)
            length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")
            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
        end

        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(distributions) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
        return new(name, parents, distributions, samples, discretization)
    end
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distributions::Dict{Vector{Symbol},D}
) where {D<:Distribution}

    samples = Dict{Vector{Symbol},DataFrame}()
    discretization = ApproximatedDiscretization()
    ContinuousChildNode(name, parents, distributions, samples, discretization)
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distributions::Dict{Vector{Symbol},D},
    samples::Dict{Vector{Symbol},DataFrame}
) where {D<:Distribution}

    discretizations = ApproximatedDiscretization()
    ContinuousChildNode(name, parents, distributions, samples, discretizations)
end

function ContinuousChildNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    distributions::Dict{Vector{Symbol},D},
    discretizatio::ApproximatedDiscretization
) where {D<:Distribution}

    samples = Dict{Vector{Symbol},DataFrame}()
    ContinuousChildNode(name, parents, distributions, samples, discretizatio)
end

function get_randomvariable(node::ContinuousChildNode, evidence::Vector{Symbol})
    node_keys = keys(node.distributions) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.distributions)]) && error("evidence $evidence does not contain all the parents of the ContinuousChildNode $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return RandomVariable(node.distributions[key], node.name)
end

function Base.isequal(node1::ContinuousChildNode, node2::ContinuousChildNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && keys(node1.distributions) == keys(node2.distributions) && node1.discretization.intervals == node2.discretization.intervals && node1.discretization.sigma == node2.discretization.sigma
end

function Base.hash(node::ContinuousChildNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.distributions, h)
    h = hash(node.samples, h)
    h = hash(node.discretization.intervals, h)
    h = hash(node.discretization.sigma, h)
    return h
end

``` DiscreteChildNode
```
mutable struct DiscreteChildNode <: DiscreteNode
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

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        for (key, val) in states
            verify_probabilities(val)
            verify_parameters(val, parameters)
            length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")
            any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
        end

        node_states = [keys(s) for s in values(states)]
        if length(reduce(intersect, node_states)) != length(reduce(union, node_states))
            error("node $name: non-coherent definition of nodes states")
        end

        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
        length(discrete_parents_combination) != length(states) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")

        return new(name, parents, states, covs, samples, parameters)
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


function Base.isequal(node1::DiscreteChildNode, node2::DiscreteChildNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.states == node2.states && node1.parameters == node2.parameters
end

function Base.hash(node::DiscreteChildNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.states, h)
    h = hash(node.covs, h)
    h = hash(node.samples, h)
    h = hash(node.parameters, h)

    return h
end

const global ChildNode = Union{DiscreteChildNode,ContinuousChildNode}