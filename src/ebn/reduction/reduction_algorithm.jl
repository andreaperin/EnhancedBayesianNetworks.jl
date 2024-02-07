function _is_reducible(net::EnhancedBayesianNetwork)
    try
        reduce(net)
    catch
        return false
    end
    return true
end

function reduce(net::EnhancedBayesianNetwork)
    rbn = deepcopy(net)
    continuous_nodes = filter(x -> isa(x, ContinuousNode), rbn.nodes)
    continuous_notfun_nodes = filter(x -> !isa(x, ContinuousFunctionalNode), continuous_nodes)
    r_dag_nodes = copy(rbn.nodes)
    r_dag = rbn.dag
    while !isempty(continuous_notfun_nodes)
        starting_node = continuous_notfun_nodes[findmin(map(x -> length(get_parents(rbn, x)), continuous_notfun_nodes))[2]]
        starting_node_index = findall(isequal.(repeat([starting_node], length(r_dag_nodes)), r_dag_nodes))[1]
        children = get_children(rbn, starting_node)
        r_dag = EnhancedBayesianNetworks._reduce_continuousnode(r_dag, starting_node_index)
        for child in children
            for parent in child.parents
                parent.name == starting_node.name && deleteat!(child.parents, findall(x -> x == parent, child.parents))
            end
            if !isa(starting_node, ContinuousRootNode)
                starting_node_parents = filter(x -> isa(x, DiscreteNode), starting_node.parents)
                for s in starting_node_parents
                    .!any(isequal.(repeat([s], length(child.parents)), child.parents)) && push!(child.parents, s)
                end
            end
        end
        r_dag_nodes = deleteat!(r_dag_nodes, starting_node_index)
        deleteat!(continuous_notfun_nodes, findall(x -> x == starting_node, continuous_notfun_nodes))
    end

    ordered_rdag, ordered_rnodes, ordered_rname_to_index = _topological_ordered_dag(r_dag_nodes)

    new_net = EnhancedBayesianNetwork(ordered_rdag, ordered_rnodes, ordered_rname_to_index)

    return new_net
end

function _reduce_continuousnode(dag::SimpleDiGraph, node_index::Int)
    r_dag = deepcopy(dag)
    child_indices = r_dag.fadjlist[node_index]
    for child in child_indices
        r_dag = _invert_link_nodes(r_dag, node_index, child)
    end
    _remove_barren_node(r_dag, node_index)
end

### NODEs and DAGs operation
function _invert_link_dag(dag::SimpleDiGraph, parent_index::Int, child_index::Int)
    new_dag = deepcopy(dag)
    child_index ∉ dag.fadjlist[parent_index] && error("Invalid dag-link to be inverted")
    rem_edge!(new_dag, parent_index, child_index)
    add_edge!(new_dag, child_index, parent_index)
    is_cyclic(new_dag) ? error("Cyclic dag error") : return new_dag
end

function _invert_link_nodes(dag::SimpleDiGraph, parent_index::Int, child_index::Int)
    parents_pr = dag.badjlist[parent_index]
    parents_ch = setdiff(dag.badjlist[child_index], parent_index)
    new_dag = _invert_link_dag(dag, parent_index, child_index)
    [add_edge!(new_dag, i, parent_index) for i in parents_ch]
    [add_edge!(new_dag, j, child_index) for j in parents_pr]
    is_cyclic(new_dag) ? error("Cyclic dag error") : new_dag = new_dag
    return new_dag
end

function _remove_barren_node(dag::SimpleDiGraph, node_index::Int)
    !isempty(dag.fadjlist[node_index]) && error("node to be eliminated must be a barren node")
    for i in deepcopy(dag.badjlist[node_index])
        rem_edge!(dag, i, node_index)
    end
    deleteat!(dag.badjlist, node_index)
    new_badjlist = Vector{Vector{Int64}}()
    for i in dag.badjlist
        push!(new_badjlist, i .- Int.(i .> node_index))
    end
    deleteat!(dag.fadjlist, node_index)
    new_fadjlist = Vector{Vector{Int64}}()
    for i in dag.fadjlist
        push!(new_fadjlist, i .- Int.(i .> node_index))
    end
    return SimpleDiGraph(dag.ne, new_fadjlist, new_badjlist)
end