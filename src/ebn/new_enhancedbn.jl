mutable struct EnhancedBayesianNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology_dict::Dict
    adj_matrix::SparseMatrixCSC

    function EnhancedBayesianNetwork(nodes::AbstractVector{<:AbstractNode}, topology_dict::Dict, adj_matrix::SparseMatrixCSC)
        nodes_names = map(i -> i.name, nodes)
        if nodes_names != unique(nodes_names)
            error("network nodes names must be unique")
        end
        discrete_nodes = filter(x -> isa(x, DiscreteNode) && !isa(x, FunctionalNode), nodes)
        states_list = mapreduce(i -> EnhancedBayesianNetworks._get_states(i), vcat, discrete_nodes)
        if states_list != unique(states_list)
            error("network nodes states must be unique")
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

function add_child!(net::EnhancedBayesianNetwork, par::Symbol, ch::Symbol)
    index_par = net.topology_dict[par]
    index_ch = net.topology_dict[ch]
    nodes = net.nodes
    par_node = first(filter(n -> n.name == par, nodes))
    ch_node = first(filter(n -> n.name == ch, nodes))
    ## No recursion in BayesianNetworks
    if par == ch
        error("Recursion on the same node is not allowed in EnhancedBayesianNetworks")
    end
    ## Check children of functional node needs to be functional Nodes
    if isa(par_node, FunctionalNode) && !isa(ch_node, FunctionalNode)
        error("Functional node $par can have only functional children, and $ch is not")
    end
    ## Root Nodes cannot be childrens
    if isa(ch_node, RootNode)
        error("root node $ch cannot have parents")
    end
    ## Check parents is in the scenarios with all its states
    if isa(ch_node, ChildNode)
        par_states = _get_states(par_node)
        scenarios = collect(keys(ch_node.states))
        is_present = map(scenario -> any((par_states) .∈ [scenario]), scenarios)
        if !all(is_present)
            wrong_scenarios = scenarios[.!is_present]
            error("child node $ch has scenarios $wrong_scenarios, that do not contain any of $par_states from its parent $par")
        end
    end
    net.adj_matrix[index_par, index_ch] = 1
    return nothing
end

function add_child!(net::EnhancedBayesianNetwork, par_index::Int64, ch_index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    par = reverse_dict[par_index]
    ch = reverse_dict[ch_index]
    add_child!(net, par, ch)
end

function add_child!(net::EnhancedBayesianNetwork, par_node::AbstractNode, ch_node::AbstractNode)
    par = par_node.name
    ch = ch_node.name
    add_child!(net, par, ch)
end

function order_net!(net::EnhancedBayesianNetwork)
    n = net.adj_matrix.n
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    all_nodes = range(1, n)
    root_indices = findall(map(col -> all(col .== 0), eachcol(net.adj_matrix)))
    root_nodes = AbstractVector{AbstractNode}(map(x -> first(filter(j -> j.name == reverse_dict[x], net.nodes)), root_indices))
    to_be_classified = setdiff(all_nodes, root_indices)
    while !isempty(to_be_classified)
        par_list = map(r -> net.adj_matrix[:, r].nzind, to_be_classified)
        new_root_indices = findall(map(p -> all(p .∈ [root_indices]), par_list))
        new_root = map(i -> to_be_classified[i], new_root_indices)
        append!(root_indices, new_root)
        new_root_nodes = map(x -> first(filter(j -> j.name == reverse_dict[x], net.nodes)), new_root)
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

function _verify_net(net::EnhancedBayesianNetwork)
    nodes2check = filter(x -> !isa(x, RootNode), net.nodes)
    map(n -> _verify_node(n, get_parents(net, net.topology_dict[n.name])[3]), nodes2check)
    return nothing
end

# function _get_edges(adj_matrix::SparseMatrixCSC)
#     n = size(adj_matrix)
#     edge_list = Vector{Tuple{Int64,Int64}}()
#     for i in range(1, n[1])
#         for j in range(1, n[2])
#             if adj_matrix[i, j] != 0
#                 push!(edge_list, (i, j))
#             end
#         end
#     end
#     return edge_list
# end

function get_parents(net::EnhancedBayesianNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    indices = net.adj_matrix[:, index].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function get_parents(net::EnhancedBayesianNetwork, name::Symbol)
    index = net.topology_dict[name]
    get_parents(net, index)
end

function get_parents(net::EnhancedBayesianNetwork, node::AbstractNode)
    index = net.topology_dict[node.name]
    get_parents(net, index)
end

function get_children(net::EnhancedBayesianNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    indices = net.adj_matrix[index, :].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function get_children(net::EnhancedBayesianNetwork, name::Symbol)
    index = net.topology_dict[name]
    get_children(net, index)
end

function get_children(net::EnhancedBayesianNetwork, node::AbstractNode)
    index = net.topology_dict[node.name]
    get_children(net, index)
end

function markov_blanket(net::EnhancedBayesianNetwork, index::Int64)
    blanket = []
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    for child in get_children(net, index)[1]
        append!(blanket, get_parents(net, child)[1])
        push!(blanket, child)
    end
    append!(blanket, get_parents(net, index)[1])
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
    fun = (a, b) -> unique(vcat(b, mapreduce(x -> filter(x -> isa(x, ContinuousNode), markov_blanket(a, node)[3]), vcat, b)))
    list = [node]
    new_list = fun(net, list)
    while !issetequal(list, new_list)
        list = new_list
        new_list = fun(net, new_list)
    end
    return new_list
end

function markov_envelope(net)
    cont_nodes = filter(x -> isa(x, ContinuousNode), net.nodes)
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