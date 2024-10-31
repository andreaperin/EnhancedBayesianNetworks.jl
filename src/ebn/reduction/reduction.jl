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
