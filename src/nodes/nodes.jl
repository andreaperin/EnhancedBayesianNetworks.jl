abstract type AbstractNode end
abstract type DiscreteNode <: AbstractNode end
abstract type ContinuousNode <: AbstractNode end

const AbstractSimulation = Union{AbstractMonteCarlo,LineSampling,SubSetSimulation}

include("../util/node_verification.jl")
include("root.jl")
include("child.jl")
include("functional.jl")

function discrete_ancestors(node::AbstractNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)

    if isempty(continuous_parents)
        return discrete_parents
    end

    return unique([discrete_parents..., mapreduce(discrete_ancestors, vcat, continuous_parents)...])
end

function discrete_ancestors(_::RootNode)
    return AbstractNode[]
end

function state_combinations(node::AbstractNode)
    par = discrete_ancestors(node)
    discrete_parents = filter(x -> isa(x, DiscreteNode), par)
    discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
    discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
    return vec(discrete_parents_combination)
end

function state_combinations(_::RootNode)
    return AbstractNode[]
end


