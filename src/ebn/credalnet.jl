struct CredalNetwork <: ProbabilisticGraphicalModel
    dag::SimpleDiGraph
    nodes::Vector{<:DiscreteNode}
    name_to_index::Dict{Symbol,Int}

    function CredalNetwork(dag::DiGraph, nodes::Vector{<:AbstractNode}, name_to_index::Dict{Symbol,Int})
        if any([isa(x, FunctionalNode) for x in nodes])
            error("Network needs to be evaluated first")
        else
            if any([!isa(x, DiscreteNode) for x in nodes])
                error("Credal Network allows discrete node only!")
            elseif all(.!_is_imprecise.(nodes))
                error("When all nodes are precise use BayesNetwork structure")
            else
                nodes = Vector{DiscreteNode}(nodes)
            end
        end
        new(dag, nodes, name_to_index)
    end
end

function CredalNetwork(nodes::Vector{<:AbstractNode})
    ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
    CredalNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
end