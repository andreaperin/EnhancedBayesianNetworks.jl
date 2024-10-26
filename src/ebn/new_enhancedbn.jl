mutable struct Network
    topology_dict::Dict
    adj_matrix::SparseMatrixCSC
end

function Network(nodes::AbstractVector)
    n = length(nodes)
    topology_dict = Dict()
    for (i, n) in enumerate(nodes)
        topology_dict[n.name] = i
    end
    adj_matrix = sparse(zeros(n, n))
    return Network(topology_dict, adj_matrix)
end

function add_child!(net::Network, nodes::AbstractVector, par::Symbol, ch::Symbol)
    index_par = net.topology_dict[par]
    index_ch = net.topology_dict[ch]
    net.adj_matrix[index_par, index_ch] = 1
    return nothing
end

function order_net!(net::Network)
    n = net.adj_matrix.n
    all_nodes = range(1, n)
    root_indices = findall(map(col -> all(col .== 0), eachcol(net.adj_matrix)))
    to_be_classified = setdiff(all_nodes, root_indices)
    while !isempty(to_be_classified)
        par_list = map(r -> net.adj_matrix[:, r].nzind, to_be_classified)
        new_root_indices = findall(map(p -> all(p .∈ [root_indices]), par_list))
        new_root = map(i -> to_be_classified[i], new_root_indices)
        append!(root_indices, new_root)
        to_be_classified = setdiff(to_be_classified, new_root)
    end
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
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
    return nothing
end
