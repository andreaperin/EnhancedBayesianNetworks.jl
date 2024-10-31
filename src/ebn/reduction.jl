# function _is_reducible(dag_::SimpleDiGraph, index::Int)
#     dag = deepcopy(dag_)
#     try
#         dag = _reduce_dag_single(dag, index)
#     catch
#         return false
#     end
#     return true
# end

# function _reduce_dag_single(dag::SimpleDiGraph, index::Int)
#     child_indices = dag.fadjlist[index]
#     for child in child_indices
#         dag = _invert_link(dag, index, child)
#     end
#     _remove_barren_node(dag, index)
# end

# function _invert_link(dag_::SimpleDiGraph, parent::Int, child::Int)
#     dag = deepcopy(dag_)
#     granparents = dag.badjlist[parent]
#     other_parents = setdiff(dag.badjlist[child], parent)
#     dag = _invert_link_simple(dag, parent, child)
#     [add_edge!(dag, i, parent) for i in other_parents]

#     [add_edge!(dag, i, child) for i in granparents]
#     is_cyclic(dag) ? error("Cyclic dag error") : return dag
# end

# function _invert_link_simple(dag_::SimpleDiGraph, parent::Int, child::Int)
#     dag = deepcopy(dag_)
#     child ∉ dag.fadjlist[parent] && error("Invalid dag-link to invert")
#     rem_edge!(dag, parent, child)
#     add_edge!(dag, child, parent)
#     is_cyclic(dag) ? error("Cyclic dag error") : return dag
# end

# function _remove_barren_node(dag::SimpleDiGraph, index::Int)
#     !isempty(dag.fadjlist[index]) && error("Cannot eliminate a not-barren node")

#     [rem_edge!(dag, i, index) for i in deepcopy(dag.badjlist[index])]
#     deleteat!(dag.badjlist, index)

#     new_badjlist = Vector{Vector{Int64}}()
#     [push!(new_badjlist, i .- Int.(i .> index)) for i in dag.badjlist]
#     deleteat!(dag.fadjlist, index)

#     new_fadjlist = Vector{Vector{Int64}}()
#     [push!(new_fadjlist, i .- Int.(i .> index)) for i in dag.fadjlist]

#     return SimpleDiGraph(dag.ne, new_fadjlist, new_badjlist)
# end

function _is_cyclic_dfs(adj_matrix)
    n = size(adj_matrix, 1)  # Number of nodes
    visited = fill(false, n)
    recStack = fill(false, n)
    # Helper function to perform DFS and detect cycles
    function dfs(v)
        visited[v] = true
        recStack[v] = true
        # Check neighbors
        for neighbor in 1:n
            if adj_matrix[v, neighbor] != 0  # there's an edge from v to neighbor
                if !visited[neighbor]  # If neighbor hasn't been visited, visit it
                    if dfs(neighbor)
                        return true  # Cycle detected
                    end
                elseif recStack[neighbor]  # If neighbor is in recStack, cycle detected
                    return true
                end
            end
        end
        # Remove the node from the recursion stack
        recStack[v] = false
        return false
    end
    # Check each node
    for node in 1:n
        if !visited[node]  # Only visit unvisited nodes
            if dfs(node)  # Cycle detected
                return true
            end
        end
    end
    return false  # No cycle found
end
