``` EnhancedBayesianNetwork
        
        Structure for build the Enhanced Bayesian Network from a list of nodes.
```
@auto_hash_equals struct EnhancedBayesianNetwork <: AbstractNetwork
    dag::DiGraph
    nodes::Vector{<:AbstractNode}
    name_to_index::Dict{Symbol,Int}

    function EnhancedBayesianNetwork(dag::DiGraph, nodes::Vector{<:AbstractNode}, name_to_index::Dict{Symbol,Int})
        all_states = vcat(_get_states.(filter(x -> !isa(x, DiscreteFunctionalNode) && isa(x, DiscreteNode), nodes))...)
        if unique([i.name for i in nodes]) != [i.name for i in nodes]
            error("nodes must have different names")
        end
        if unique(all_states) != all_states
            error("nodes state must have different symbols")
        end
        new(dag, nodes, name_to_index)
    end
end

function EnhancedBayesianNetwork(nodes::Vector{<:AbstractNode})
    ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
    ebn = EnhancedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
    return ebn
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

function get_children(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in outneighbors(ebn.dag, i)]
end

function get_parents(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in unique(inneighbors(ebn.dag, i))]
end

function get_neighbors(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in unique(append!(inneighbors(ebn.dag, i), outneighbors(ebn.dag, i)))]
end

## Returns a Set of AbstractNode representing the Markov Blanket for the choosen node
function markov_blanket(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    blanket = AbstractNode[]
    for child in get_children(ebn, node)
        append!(blanket, get_parents(ebn, child))
        push!(blanket, child)
    end
    append!(blanket, get_parents(ebn, node))
    return unique(setdiff(blanket, [node]))
end

function markov_envelope(ebn)
    Xm_groups = map(x -> _markov_envelope_continuous_nodes_group(ebn, x), filter(x -> isa(x, ContinuousNode), ebn.nodes))
    markov_envelopes = unique.(mapreduce.(x -> push!(markov_blanket(ebn, x), x), vcat, Xm_groups))
    # check when a vector is included into another
    sorted_envelopes = sort(markov_envelopes, by=length)
    final_envelopes = []
    for envelope in sorted_envelopes
        to_compare = filter(x -> x != envelope, sorted_envelopes)
        if any(map(x -> all(envelope .∈ [x]), to_compare))
            filter!(x -> x != envelope, sorted_envelopes)
        else
            push!(final_envelopes, envelope)
        end
    end
    return final_envelopes
end

function _markov_envelope_continuous_nodes_group(ebn, node)
    f = (ebn, nodes) -> unique(vcat(nodes, mapreduce(x -> filter(x -> isa(x, ContinuousNode), markov_blanket(ebn, node)), vcat, nodes)))

    list = [node]
    new_list = f(ebn, list)
    while !issetequal(list, new_list)
        list = new_list
        new_list = f(ebn, new_list)
    end
    return new_list
end