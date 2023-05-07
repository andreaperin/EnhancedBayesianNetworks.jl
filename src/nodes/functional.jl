const AbstractSimulation = Union{MonteCarlo,LineSampling,SubSetSimulation}
struct DiscreteFunctionalNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    models::OrderedDict{Vector{Symbol},Vector{M}} where {M<:UQModel}
    performances::OrderedDict{Vector{Symbol},Function}
    simulations::OrderedDict{Vector{Symbol},S} where {S<:AbstractSimulation}

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

        return new(name, parents, models, performances, simulations)
    end
end

##TODO this function and add a test!
function get_models(node::DiscreteFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    all(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the FunctionalNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return node.models[node_key]
end

function get_performance(node::DiscreteFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    all(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the FunctionalNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return node.performances[node_key]
end

function get_simulation(node::DiscreteFunctionalNode, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    all(node.parents .∉ [[x[2] for x in evidence]]) && error("evidence does not contain any parents of the FunctionalNode")
    node_key = Symbol[]
    for parent in node.parents
        append!(node_key, [e[1] for e in evidence if e[2] == parent])
    end
    return node.simulations[node_key]
end

struct StructuralReliabilityProblem
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    performance::Function
    simulation::AbstractSimulation
end

mutable struct StructuralReliabilityProblemNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::OrderedDict{Vector{Symbol},StructuralReliabilityProblem}
end