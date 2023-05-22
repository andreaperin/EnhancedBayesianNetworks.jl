struct EnhancedBayesianNetwork <: ProbabilisticGraphicalModel
    dag::DiGraph
    nodes::Vector{<:AbstractNode}
    name_to_index::Dict{Symbol,Int}

    function EnhancedBayesianNetwork(dag::DiGraph, nodes::Vector{<:AbstractNode}, name_to_index::Dict{Symbol,Int})
        all_states = vcat(_get_states.(filter(x -> !isa(x, DiscreteFunctionalNode) && isa(x, DiscreteNode), nodes))...)
        unique(all_states) != all_states ? error("nodes state must have different symbols") :
        new(dag, nodes, name_to_index)
    end
end

function EnhancedBayesianNetwork(nodes_::Vector{<:AbstractNode})
    nodes = deepcopy(nodes_)
    ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
    ebn = EnhancedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)

    continuous_nodes = filter!(j -> !isa(j, FunctionalNode), (filter!(x -> isa(x, ContinuousNode), nodes)))
    a = isempty.([i.intervals for i in continuous_nodes])
    evidence_node = continuous_nodes[.!a]
    while !isempty(evidence_node)
        if isa(evidence_node[1], RootNode)
            nodes = _discretize_node(ebn, evidence_node[1], evidence_node[1].intervals)
            ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
            ebn = EnhancedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
        elseif isa(evidence_node[1], StandardNode)
            nodes = _discretize_node(ebn, evidence_node[1], evidence_node[1].intervals, evidence_node[1].sigma)
            ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
            ebn = EnhancedBayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
        end
        popfirst!(evidence_node)
    end
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

function plot(ebn::EnhancedBayesianNetwork)
    graphplot(
        ebn.dag,
        names=[i.name for i in ebn.nodes],
        # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), ebn.nodes),
        font_size=10,
        node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, ebn.nodes),
        markercolor=map(x -> isa(x, DiscreteFunctionalNode) ? "lightgreen" : "orange", ebn.nodes),
        linecolor=:darkgrey,
    )
end

function get_children(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in outneighbors(ebn.dag, i)]
end

function get_parents(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in inneighbors(ebn.dag, i)]
end

function get_neighbors(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in append!(inneighbors(ebn.dag, i), outneighbors(ebn.dag, i))]
end

## Returns a Set of AbstractNode representing the Markov Blanket for the choosen node
function markov_blanket(ebn::EnhancedBayesianNetwork, node::N) where {N<:AbstractNode}
    blanket = AbstractNode[]
    for child in get_children(ebn, node)
        append!(blanket, get_parents(ebn, child))
        push!(blanket, child)
    end
    append!(blanket, get_parents(ebn, node))
    return unique(setdiff(Set(blanket), Set([node])))
end

function markov_envelope(ebn::EnhancedBayesianNetwork)
    continuous_nodes = filter(x -> isa(x, ContinuousNode), ebn.nodes)
    groups = []
    for node in continuous_nodes
        new_continuous_nodes = filter(x -> isa(x, ContinuousNode), markov_blanket(ebn, node)) |> collect
        isempty(new_continuous_nodes) ? group = [node] : group = vcat(node, new_continuous_nodes)
        while !isempty(new_continuous_nodes)
            blanket_i = filter(x -> isa(x, ContinuousNode), markov_blanket(ebn, new_continuous_nodes[1]))
            popfirst!(new_continuous_nodes)
            new_continuous_nodes_i = setdiff(blanket_i, new_continuous_nodes) |> collect
            vcat(new_continuous_nodes, new_continuous_nodes_i)
            vcat(group, collect(setdiff(blanket_i, group)))
        end
        push!(groups, group)
    end
    envelope = []
    groups = unique(Set.(groups))
    longest_set = groups[findmax(length.(groups))[2]]
    for x in groups
        if all(x .∈ [longest_set]) && x != longest_set
            deleteat!(groups, findall(i -> i == x, groups))
        end
    end
    for group in groups
        group = group |> collect
        all_blankets = markov_blanket.(repeat([ebn], length(group)), group)
        single_envelope = unique(vcat(unique(Iterators.flatten(all_blankets)), group))
        push!(envelope, single_envelope)
    end
    return envelope
end

function _create_ebn_from_envelope(ebn::EnhancedBayesianNetwork, envelope::Vector{<:AbstractNode})
    nodes = Vector{AbstractNode}()
    for node in envelope
        !all(get_parents(ebn, node) .∈ [envelope]) && append!(nodes, get_parents(ebn, node))
        push!(nodes, node)
    end
    EnhancedBayesianNetwork(unique!(nodes))
end

##TODO test
function _get_node_given_state(ebn::EnhancedBayesianNetwork, state::Symbol)
    nodes = filter(x -> !isa(x, DiscreteFunctionalNode) && isa(x, DiscreteNode), ebn.nodes)
    [node for node in nodes if state ∈ _get_states(node)][1]
end
