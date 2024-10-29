mutable struct Network
    nodes::AbstractVector{<:AbstractNode}
    topology_dict::Dict
    adj_matrix::SparseMatrixCSC
end

function Network(nodes::AbstractVector{<:AbstractNode})
    n = length(nodes)
    topology_dict = Dict()
    for (i, n) in enumerate(nodes)
        topology_dict[n.name] = i
    end
    adj_matrix = sparse(zeros(n, n))
    return Network(nodes, topology_dict, adj_matrix)
end

function add_child!(net::Network, par::Symbol, ch::Symbol)
    index_par = net.topology_dict[par]
    index_ch = net.topology_dict[ch]
    nodes = net.nodes
    par_node = first(filter(n -> n.name == par, nodes))
    ch_node = first(filter(n -> n.name == ch, nodes))
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

function order_net!(net::Network)
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

function _verify_net(net::Network)
    nodes2check = filter(x -> !isa(x, RootNode), net.nodes)
    map(n -> _verify_node(n, get_parents(net, net.topology_dict[n.name])[3]), nodes2check)
    return nothing
end

function _verify_node(node::ChildNode, parents::AbstractVector)
    ## Check scenarios coherence for non functional nodes
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    discrete_parents_combination = Iterators.product(_get_states.(discrete_parents)...)
    discrete_parents_combination = vec(map(t -> [t...], discrete_parents_combination))
    discrete_parents_combination_set = map(x -> Set(x), discrete_parents_combination)
    scenarios_set = map(x -> Set(x), _get_scenarios(node))
    if !issetequal(discrete_parents_combination, _get_scenarios(node))
        is_present = map(x -> x ∈ scenarios_set, discrete_parents_combination_set)
        missing_parents_combinations = discrete_parents_combination_set[.!is_present]
        if !isempty(missing_parents_combinations)
            error("parents combinations $missing_parents_combinations, are missing in node $(node.name) defined scenarios $scenarios_set")
        end
    end
    return nothing
end

function _verify_node(node::FunctionalNode, parents::AbstractVector)
    continuous_parents = filter(x -> isa(x, ContinuousNode), parents)
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents)
    if isempty(continuous_parents)
        error("functional nodes $(node.name) must have at least one continuous parent")
    end
    have_no_parameter = map(x -> isempty(x.parameters), discrete_parents)
    no_parameters_nodes = discrete_parents[have_no_parameter]
    ## Functional Node must have discrete parents with a defined parameters argument
    if !isempty(no_parameters_nodes)
        no_parameters_nodes_name = [i.name for i in no_parameters_nodes]
        error("node/s $no_parameters_nodes_name are discrete and parents of the functional node $(node.name), therefore a parameter argument must be defined")
    end
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

function get_parents(net::Network, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    indices = net.adj_matrix[:, index].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function get_parents(net::Network, name::Symbol)
    index = net.topology_dict[name]
    get_parents(net, index)
end

function get_parents(net::Network, node::AbstractNode)
    index = net.topology_dict[node.name]
    get_parents(net, index)
end