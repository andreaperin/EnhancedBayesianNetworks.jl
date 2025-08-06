@auto_hash_equals mutable struct BayesianNetwork2be
    nodes::AbstractVector{Symbol}
    topology_dict::Dict
    adj_matrix::SparseMatrixCSC

    function BayesianNetwork2be(nodes::AbstractVector{Symbol}, topology_dict::Dict, adj_matrix::SparseMatrixCSC)
        nodes_names = nodes
        if !allunique(nodes_names)
            error("network nodes names must be unique")
        end
        new(nodes, topology_dict, adj_matrix)
    end
end

function BayesianNetwork2be(nodes::AbstractVector{Symbol})
    n = length(nodes)
    topology_dict = Dict()
    for (i, n) in enumerate(nodes)
        topology_dict[n] = i
    end
    adj_matrix = sparse(zeros(n, n))
    return BayesianNetwork2be(nodes, topology_dict, adj_matrix)
end

BayesianNetwork(nodes::AbstractVector{Symbol}) = BayesianNetwork2be(nodes)

function add_child!(net::BayesianNetwork2be, par::Symbol, ch::Symbol)
    index_par = net.topology_dict[par]
    index_ch = net.topology_dict[ch]
    nodes = net.nodes
    par_node = first(filter(n -> n == par, nodes))
    ch_node = first(filter(n -> n == ch, nodes))
    _verify_no_recursion(par_node, ch_node)
    net.adj_matrix[index_par, index_ch] = 1
    return nothing
end

function add_child!(net::BayesianNetwork2be, par_index::Int64, ch_index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    par = reverse_dict[par_index]
    ch = reverse_dict[ch_index]
    add_child!(net, par, ch)
end

function order!(net::BayesianNetwork2be)
    if _is_cyclic_dfs(net.adj_matrix)
        error("network is cyclic!")
    end
    n = net.adj_matrix.n
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    all_nodes = range(1, n)
    root_indices = findall(map(col -> all(col .== 0), eachcol(net.adj_matrix)))
    root_nodes = AbstractVector{Symbol}(map(x -> first(filter(j -> j == reverse_dict[x], net.nodes)), root_indices))
    to_be_classified = setdiff(all_nodes, root_indices)
    while !isempty(to_be_classified)
        par_list = map(r -> net.adj_matrix[:, r].nzind, to_be_classified)
        new_root_indices = findall(map(p -> all(p .∈ [root_indices]), par_list))
        new_root = map(i -> to_be_classified[i], new_root_indices)
        append!(root_indices, new_root)
        new_root_nodes = map(x -> first(filter(j -> j == reverse_dict[x], net.nodes)), new_root)
        append!(root_nodes, new_root_nodes)
        to_be_classified = setdiff(to_be_classified, new_root)
    end

    ordered_topology_dict = Dict(map(i -> (reverse_dict[i[2]], i[1]), enumerate(root_indices)))
    conversion = Dict(map(i -> (i[2], i[1]), enumerate(root_indices)))
    ordered_matrix = sparse(zeros(n, n))
    for i in range(1, n)
        for j in range(1, n)
            if net.adj_matrix[i, j] == 1
                ordered_matrix[conversion[i], conversion[j]] = 1
            end
        end
    end

    net.adj_matrix = ordered_matrix
    net.topology_dict = ordered_topology_dict
    net.nodes = root_nodes
    return nothing
end

function parents(net::BayesianNetwork2be, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    indices = net.adj_matrix[:, index].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x ∈ names, net.nodes)
    return indices, nodes
end

function parents(net::BayesianNetwork2be, name::Symbol)
    index = net.topology_dict[name]
    parents(net, index)
end

function children(net::BayesianNetwork2be, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    indices = net.adj_matrix[index, :].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x ∈ names, net.nodes)
    return indices, nodes
end

function children(net::BayesianNetwork2be, name::Symbol)
    index = net.topology_dict[name]
    children(net, index)
end