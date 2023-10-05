abstract type AbstractNode end
abstract type DiscreteNode <: AbstractNode end
abstract type ContinuousNode <: AbstractNode end

const AbstractSimulation = Union{AbstractMonteCarlo,LineSampling,SubSetSimulation}

include("../util/node_verification.jl")
include("root.jl")
include("child.jl")
include("functional.jl")
include("structuralreliabilityproblemnode.jl")

function get_discrete_ancestors(node::AbstractNode)
    discrete_nodes = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_nodes = filter(x -> isa(x, ContinuousNode), node.parents)
    while !isempty(continuous_nodes)
        if isa(continuous_nodes[1], RootNode)
            popfirst!(continuous_nodes)
        else
            new_discrete_nodes = filter(x -> isa(x, DiscreteNode), continuous_nodes[1].parents)
            append!(discrete_nodes, new_discrete_nodes)
            new_continuous_nodes = filter(x -> isa(x, ContinuousNode), continuous_nodes[1].parents)
            append!(continuous_nodes, new_continuous_nodes)
            popfirst!(continuous_nodes)
        end
    end
    return discrete_nodes
end

function get_states_combinantions(node::AbstractNode)
    if isa(node, RootNode)
        discrete_parents_combination = []
    else
        par = get_discrete_ancestors(node)
        discrete_parents = filter(x -> isa(x, DiscreteNode), par)
        discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
        discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
    end
    return vec(discrete_parents_combination)
end

