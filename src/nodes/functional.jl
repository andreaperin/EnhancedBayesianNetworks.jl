mutable struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::Dict{Vector{Symbol},Vector{UQModel}}
    simulations::Union{Dict{Vector{Symbol},S},Nothing} where {S<:AbstractSimulation}
    samples::Dict{Vector{Symbol},Any}
    parameters::Dict{Symbol,Vector{Parameter}}

    function ContinuousFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Dict{Vector{Symbol},Vector{M}},
        simulations::Union{Dict{Vector{Symbol},S},Nothing},
        parameters::Dict{Symbol,Vector{Parameter}}
    ) where {M<:UQModel,S<:AbstractSimulation}

        discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
        verify_functionalnode_parents(parents)

        for i in [models, simulations]
            for (key, _) in i
                length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")

                any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
            end
            discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
            discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
            length(discrete_parents_combination) != length(i) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
        end
        samples = Dict{Vector{Symbol},Any}()
        return new(name, parents, models, simulations, samples, parameters)
    end
end

function ContinuousFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Dict{Vector{Symbol},Vector{M}},
    simulations::Union{Dict{Vector{Symbol},S},Nothing}
) where {M<:UQModel,S<:AbstractSimulation}
    ContinuousFunctionalNode(name, parents, models, simulations, Dict{Symbol,Vector{Parameter}}())
end


## Get all the parents random variable if the evidence gives uniques random variables 
function get_randomvariable(node::ContinuousFunctionalNode, evidence::Vector{Symbol})
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    return mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents)
end

function get_models(node::ContinuousFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.models) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.models)]) && error("evidence $evidence does not contain all the parents of the ContinuousChildNode $name")
    key = node_keys[findfirst([issubset(evidence, i) for i in node_keys])]
    return node.models[key]
end

function get_simulation(node::ContinuousFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.simulations) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.simulations)]) && error("evidence $evidence does not contain all the parents of the ContinuousChildNode $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return node.simulations[key]
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
    models::Dict{Vector{Symbol},Vector{UQModel}}
    performances::Dict{Vector{Symbol},Function}
    simulations::Union{Dict{Vector{Symbol},S},Nothing} where {S<:AbstractSimulation}
    pf::Dict{Vector{Symbol},Real}
    cov::Dict{Vector{Symbol},Real}
    samples::Dict{Vector{Symbol},Any}
    parameters::Dict{Symbol,Vector{Parameter}}

    function DiscreteFunctionalNode(
        name::Symbol,
        parents::Vector{<:AbstractNode},
        models::Dict{Vector{Symbol},Vector{M}},
        performances::Dict{Vector{Symbol},Function},
        simulations::Union{Dict{Vector{Symbol},S},Nothing},
        parameters::Dict{Symbol,Vector{Parameter}}
    ) where {M<:UQModel,S<:AbstractSimulation}

        if isempty(filter(x -> isa(x, FunctionalNode), parents))
            discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
            verify_functionalnode_parents(parents)

            for i in [models, performances, simulations]
                for (key, _) in i
                    length(discrete_parents) != length(key) && error("In node $name, defined parents states differ from number of its discrete parents")

                    any([k ∉ _get_states(discrete_parents[i]) for (i, k) in enumerate(key)]) && error("In node $name, defined parents states are not coherent with its discrete parents states")
                end

                discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
                discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
                length(discrete_parents_combination) != length(i) && error("In node $name, defined combinations are not equal to the theorical discrete parents combinations: $discrete_parents_combination")
            end
        end
        pf = Dict{Vector{Symbol},Real}()
        cov = Dict{Vector{Symbol},Real}()
        samples = Dict{Vector{Symbol},Any}()
        return new(name, parents, models, performances, simulations, pf, cov, samples, parameters)
    end
end

function DiscreteFunctionalNode(
    name::Symbol,
    parents::Vector{<:AbstractNode},
    models::Dict{Vector{Symbol},Vector{M}},
    performances::Dict{Vector{Symbol},Function},
    simulations::Union{Dict{Vector{Symbol},S},Nothing}
) where {M<:UQModel,S<:AbstractSimulation}
    DiscreteFunctionalNode(name, parents, models, performances, simulations, Dict{Symbol,Vector{Parameter}}())
end

# function DiscreteFunctionalNode(
#     name::Symbol,
#     parents::Vector{<:AbstractNode},
#     models::Dict{Vector{Symbol},Vector{M}},
#     performances::Dict{Vector{Symbol},Function},
# ) where {M<:UQModel}
#     DiscreteFunctionalNode(name, parents, models, performances, nothing, Dict{Symbol,Vector{Parameter}}())
# end

function get_models(node::DiscreteFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.models) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.models)]) && error("evidence $evidence does not contain all the parents of $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return node.models[key]
end

function get_performance(node::DiscreteFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.performances) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.performances)]) && error("evidence $evidence does not contain all the parents of $name")
    key = node_keys[findfirst([issubset(i, evidence) for i in node_keys])]
    return node.performances[key]
end

function get_simulation(node::DiscreteFunctionalNode, evidence::Vector{Symbol})
    node_keys = keys(node.simulations) |> collect
    name = node.name
    all(.![issubset(i, evidence) for i in keys(node.simulations)]) && error("evidence $evidence does not contain all the parents of $name")
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