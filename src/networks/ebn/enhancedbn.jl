@auto_hash_equals mutable struct EnhancedBayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology_dict::Dict
    adj_matrix::SparseMatrixCSC

    function EnhancedBayesianNetwork(nodes::AbstractVector{<:AbstractNode}, topology_dict::Dict, adj_matrix::SparseMatrixCSC)
        nodes_names = map(i -> i.name, nodes)
        if !allunique(nodes_names)
            error("network nodes names must be unique")
        end
        discrete_nodes = filter(x -> isa(x, DiscreteNode), nodes)
        if !isempty(discrete_nodes)
            states_list = mapreduce(i -> _states(i), vcat, discrete_nodes)
            if !allunique(states_list)
                error("network nodes states must be unique")
            end
        end
        new(nodes, topology_dict, adj_matrix)
    end
end

function EnhancedBayesianNetwork(nodes::AbstractVector{<:AbstractNode})
    n = length(nodes)
    topology_dict = Dict()
    for (i, n) in enumerate(nodes)
        topology_dict[n.name] = i
    end
    adj_matrix = sparse(zeros(n, n))
    return EnhancedBayesianNetwork(nodes, topology_dict, adj_matrix)
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

function markov_blanket(net::EnhancedBayesianNetwork, index::Int64)
    blanket = []
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    for child in children(net, index)[1]
        append!(blanket, parents(net, child)[1])
        push!(blanket, child)
    end
    append!(blanket, parents(net, index)[1])
    indices = unique(setdiff(blanket, [index]))
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function markov_blanket(net::EnhancedBayesianNetwork, name::Symbol)
    index = net.topology_dict[name]
    markov_blanket(net, index)
end

function markov_blanket(net::EnhancedBayesianNetwork, node::AbstractNode)
    index = net.topology_dict[node.name]
    markov_blanket(net, index)
end

function _get_markov_group(net::EnhancedBayesianNetwork, node::AbstractNode)
    fun = (ebn, n) -> unique(vcat(n, mapreduce(x -> filter(x -> isa(x, AbstractContinuousNode), markov_blanket(ebn, node)[3]), vcat, n)))
    list = [node]
    new_list = fun(net, list)
    while !issetequal(list, new_list)
        list = new_list
        new_list = fun(net, new_list)
    end
    return new_list
end

function markov_envelope(net::EnhancedBayesianNetwork)
    cont_nodes = filter(x -> isa(x, AbstractContinuousNode), net.nodes)
    Xm_groups = map(x -> _get_markov_group(net, x), cont_nodes)
    markov_envelopes = unique.(mapreduce.(x -> push!(markov_blanket(net, x)[3], x), vcat, Xm_groups))
    # check when a vector is included into another
    sorted_envelopes = sort(markov_envelopes, by=length, rev=true)
    final_envelopes = []
    while length(sorted_envelopes) >= 1
        if length(sorted_envelopes) == 1
            append!(final_envelopes, sorted_envelopes)
            popfirst!(sorted_envelopes)
        else
            envelope = first(sorted_envelopes)
            to_compare_list = sorted_envelopes[2:end]
            is_excluded = map(to_compare -> any(to_compare .∉ [envelope]), to_compare_list)
            sorted_envelopes = to_compare_list[is_excluded]
            push!(final_envelopes, envelope)
        end
    end
    return final_envelopes
end