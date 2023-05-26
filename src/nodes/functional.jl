mutable struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::OrderedDict{Vector{Symbol},Vector{UQModel}}

    function ContinuousFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::OrderedDict{Vector{Symbol},Vector{M}}
    ) where {M<:UQModel}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        verify_functionalnode_parents(parents)

        for i in [models]
            for (key, _) in i
                length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")

                any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents")
            end
            discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
            discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
            length(discrete_parents_combination) != length(i) && error("defined combinations must be equal to the discrete parents combinations")
        end

        return new(name, parents, models)
    end
end

## Get all the parents random variable if the evidence gives uniques random variables 
function get_randomvariable(node::ContinuousFunctionalNode, evidence::Vector{Symbol})
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    return mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents)
end

function get_models(node::ContinuousFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.models) |> collect
    all(.![issubset(i, evidence) for i in keys(node.models)]) && error("evidence does not contain all the parents of the ContinuousFunctionalNode")
    key = node_keys[findfirst([issubset(evidence, i) for i in node_keys])]
    return node.models[key]
end


function Base.isequal(node1::ContinuousFunctionalNode, node2::ContinuousFunctionalNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.models == node2.models
end

function Base.hash(node::ContinuousFunctionalNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.models, h)

    return h
end

mutable struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::OrderedDict{Vector{Symbol},Vector{UQModel}}
    performances::OrderedDict{Vector{Symbol},Function}
    simulations::OrderedDict{Vector{Symbol},S} where {S<:AbstractSimulation}
    pf::Dict{Vector{Symbol},Real}
    cov::Dict{Vector{Symbol},Real}
    samples::Dict{Vector{Symbol},Any}

    function DiscreteFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::OrderedDict{Vector{Symbol},Vector{M}},
        performances::OrderedDict{Vector{Symbol},Function},
        simulations::OrderedDict{Vector{Symbol},S}
    ) where {M<:UQModel,S<:AbstractSimulation}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        verify_functionalnode_parents(parents)

        for i in [models, performances, simulations]
            for (key, _) in i
                length(discrete_parents) != length(key) && error("defined parent nodes states must be equal to the number of discrete parent nodes")

                any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents")
            end

            discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
            discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
            length(discrete_parents_combination) != length(i) && error("defined combinations must be equal to the discrete parents combinations")
        end
        pf = Dict{Vector{Symbol},Real}()
        cov = Dict{Vector{Symbol},Real}()
        samples = Dict{Vector{Symbol},Any}()
        return new(name, parents, models, performances, simulations, pf, cov, samples)
    end
end

function get_models(node::DiscreteFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.models) |> collect
    all(.![issubset(i, evidence) for i in keys(node.models)]) && error("evidence does not contain all the parents of the DiscreteFunctionalNode")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return node.models[key]
end

function get_performance(node::DiscreteFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.performances) |> collect
    all(.![issubset(i, evidence) for i in keys(node.performances)]) && error("evidence does not contain all the parents of the DiscreteFunctionalNode")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return node.performances[key]
end

function get_simulation(node::DiscreteFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.simulations) |> collect
    all(.![issubset(i, evidence) for i in keys(node.simulations)]) && error("evidence does not contain all the parents of the DiscreteFunctionalNode")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return node.simulations[key]
end

function Base.isequal(node1::DiscreteFunctionalNode, node2::DiscreteFunctionalNode)
    node1.name == node2.name && issetequal(node1.parents, node2.parents) && node1.models == node2.models && node1.performances == node2.performances && node1.simulations == node2.simulations
end

function Base.hash(node::DiscreteFunctionalNode, h::UInt)
    h = hash(node.name, h)
    h = hash(node.parents, h)
    h = hash(node.models, h)
    h = hash(node.performances, h)
    h = hash(node.simulations, h)

    return h
end


const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}