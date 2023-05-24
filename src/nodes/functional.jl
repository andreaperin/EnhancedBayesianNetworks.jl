mutable struct ContinuousFunctionalNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::OrderedDict{Vector{Symbol},Vector{M}} where {M<:UQModel}

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

            discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
            discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
            length(discrete_parents_combination) != length(i) && error("defined combinations must be equal to the discrete parents combinations")
        end

        return new(name, parents, models)
    end
end

function get_randomvariable(node::ContinuousFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    check = mapreduce(n -> .!is_equal.(repeat([n], length(evidence)), [x[2] for x in evidence]), vcat, discrete_parents)
    all(check) && error("evidence does not contain any parents of the ContinuousFunctionalNode")
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    return mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents)
end

function get_models(node::ContinuousFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    check = mapreduce(n -> .!is_equal.(repeat([n], length(evidence)), [x[2] for x in evidence]), vcat, discrete_parents)
    all(check) && error("evidence does not contain any parents of the ContinuousFunctionalNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2].name == parent.name])
    end
    return node.models[node_key]
end

function is_equal(node1::ContinuousFunctionalNode, node2::ContinuousFunctionalNode)
    length(node1.parents) == length(node2.parents) && node1.name == node2.name && all(is_equal.(node1.parents, node2.parents)) && node1.models == node2.models
end

mutable struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::OrderedDict{Vector{Symbol},Vector{M}} where {M<:UQModel}
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

function get_models(node::DiscreteFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    check = mapreduce(n -> .!is_equal.(repeat([n], length(evidence)), [x[2] for x in evidence]), vcat, discrete_parents)
    all(check) && error("evidence does not contain any parents of the FunctionalNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2].name == parent.name])
    end
    return node.models[node_key]
end

function get_performance(node::DiscreteFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    check = mapreduce(n -> .!is_equal.(repeat([n], length(evidence)), [x[2] for x in evidence]), vcat, discrete_parents)
    all(check) && error("evidence does not contain any parents of the FunctionalNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2].name == parent.name])
    end
    return node.performances[node_key]
end

function get_simulation(node::DiscreteFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    check = mapreduce(n -> .!is_equal.(repeat([n], length(evidence)), [x[2] for x in evidence]), vcat, discrete_parents)
    all(check) && error("evidence does not contain any parents of the FunctionalNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2].name == parent.name])
    end
    return node.simulations[node_key]
end


function is_equal(node1::DiscreteFunctionalNode, node2::DiscreteFunctionalNode)
    length(node1.parents) == length(node2.parents) && node1.name == node2.name && all(is_equal.(node1.parents, node2.parents)) && node1.models == node2.models && node1.performances == node2.performances && node1.simulations == node2.simulations
end

const global FunctionalNode = Union{DiscreteFunctionalNode,ContinuousFunctionalNode}