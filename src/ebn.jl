abstract type ProbabilisticGraphicalModel end

struct EnhancedBayesianNetwork <: ProbabilisticGraphicalModel
    dag::DiGraph
    nodes::Vector{<:AbstractNode}
    name_to_index::Dict{Symbol,Int}
    ## With the new structure the Ebn can never be cyclic!
    # function EnhancedBayesianNetwork(dag::DiGraph, nodes::Vector{<:AbstractNode}, name_to_index::Dict{Symbol,Int})
    #     is_cyclic(dag) && error("Bayesian Networks must be noncyclic")
    #     new(dag, nodes, name_to_index)
    # end
end

function EnhancedBayesianNetwork(nodes::Vector{<:AbstractNode})
    ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
    EnhancedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
end


function _build_digraph(nodes::Vector{<:AbstractNode})
    name_to_index = Dict{Symbol,Int}()
    for (i, node) in enumerate(nodes)
        name_to_index[node.name] = i
    end
    dag = DiGraph(length(nodes))
    for node in filter(x -> !isa(x, RootNode), nodes)
        j = name_to_index[node.name]
        for p in node.parents
            i = name_to_index[p.name]
            add_edge!(dag, i, j)
        end
    end
    return dag
end

function _topological_ordered_dag(nodes::Vector{<:AbstractNode})
    dag = _build_digraph(nodes)
    ordered_vector = topological_sort_by_dfs(dag)
    n = length(nodes)
    ordered_name_to_index = Dict{Symbol,Int}()
    ordered_nodes = Vector{AbstractNode}(undef, n)
    for (new_index, old_index) in enumerate(ordered_vector)
        ordered_name_to_index[nodes[old_index].name] = new_index
        ordered_nodes[new_index] = nodes[old_index]
    end
    ordered_dag = _build_digraph(ordered_nodes)
    return ordered_dag, ordered_nodes, ordered_name_to_index
end

function plot(ebn::EnhancedBayesianNetwork)
    graphplot(
        ebn.dag,
        names=[i.name for i in ebn.nodes],
        font_size=10,
        node_shape=map(x -> isa(x, ContinuousNode) ? :ellipse : :hexagon, ebn.nodes),
        markercolor=map(x -> isa(x, ContinuousNode) ? "lightgreen" : "orange", ebn.nodes),
        linecolor=:darkgrey,
    )
end
