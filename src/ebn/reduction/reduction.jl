function _is_eliminable(net::EnhancedBayesianNetwork, node::AbstractNode)
    if !isa(node, ContinuousNode)
        error("node elimination algorithm is for continuous nodes and $(node.name) is discrete")
    end
    index = net.topology_dict[node.name]
    test_matrix = deepcopy(net.adj_matrix)
    parents = get_parents(net, index)[1]
    map(x -> test_matrix[x, index] = 0, parents)
    map(x -> test_matrix[index, x] = 1, parents)
    children = get_children(net, index)[1]
    map(x -> test_matrix[index, x] = 0, children)
    map(x -> test_matrix[x, index] = 1, children)
    !_is_cyclic_dfs(test_matrix)
end

function _is_eliminable(net::EnhancedBayesianNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    index = findfirst(x -> x.name == reverse_dict[index], net.nodes)
    _is_eliminable(net, net.nodes[index])
end

function _is_eliminable(net::EnhancedBayesianNetwork, name::Symbol)
    index = findfirst(x -> x.name == name, net.nodes)
    _is_eliminable(net, net.nodes[index])
end

function _is_cyclic_dfs(adj_matrix)
    n = size(adj_matrix, 1)  # Number of nodes
    visited = fill(false, n)
    recStack = fill(false, n)
    function dfs(v)
        visited[v] = true
        recStack[v] = true
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
        recStack[v] = false
        return false
    end
    for node in 1:n
        if !visited[node]  # Only visit unvisited nodes
            if dfs(node)  # Cycle detected
                return true
            end
        end
    end
    return false  # No cycle found
end