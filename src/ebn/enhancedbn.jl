@auto_hash_equals mutable struct EnhancedBayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology_dict::Dict
    adj_matrix::SparseMatrixCSC

    function EnhancedBayesianNetwork(nodes::AbstractVector{<:AbstractNode}, topology_dict::Dict, adj_matrix::SparseMatrixCSC)
        nodes_names = map(i -> i.name, nodes)
        if !allunique(nodes_names)
            error("network nodes names must be unique")
        end
        discrete_nodes = filter(x -> isa(x, DiscreteNode) && !isa(x, FunctionalNode), nodes)
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

function add_child!(net::AbstractNetwork, par::Symbol, ch::Symbol)
    index_par = net.topology_dict[par]
    index_ch = net.topology_dict[ch]
    nodes = net.nodes
    par_node = first(filter(n -> n.name == par, nodes))
    ch_node = first(filter(n -> n.name == ch, nodes))
    _verify_no_recursion(par_node, ch_node)
    _verify_root(par_node, ch_node)
    _verify_child(par_node, ch_node)
    _verify_functional_node(par_node, ch_node)
    net.adj_matrix[index_par, index_ch] = 1
    return nothing
end

function add_child!(net::AbstractNetwork, par_index::Int64, ch_index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    par = reverse_dict[par_index]
    ch = reverse_dict[ch_index]
    add_child!(net, par, ch)
end

function add_child!(net::AbstractNetwork, par_node::AbstractNode, ch_node::AbstractNode)
    par = par_node.name
    ch = ch_node.name
    add_child!(net, par, ch)
end

function order!(net::AbstractNetwork)
    if _is_cyclic_dfs(net.adj_matrix)
        error("network is cyclic!")
    end
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
    _verify_net(net)
    return nothing
end

function parents(net::AbstractNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    indices = net.adj_matrix[:, index].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function parents(net::AbstractNetwork, name::Symbol)
    index = net.topology_dict[name]
    parents(net, index)
end

function parents(net::AbstractNetwork, node::AbstractNode)
    index = net.topology_dict[node.name]
    parents(net, index)
end

function children(net::AbstractNetwork, index::Int64)
    reverse_dict = Dict(value => key for (key, value) in net.topology_dict)
    indices = net.adj_matrix[index, :].nzind
    names = map(x -> reverse_dict[x], indices)
    nodes = filter(x -> x.name ∈ names, net.nodes)
    return indices, names, nodes
end

function children(net::AbstractNetwork, name::Symbol)
    index = net.topology_dict[name]
    children(net, index)
end

function children(net::AbstractNetwork, node::AbstractNode)
    index = net.topology_dict[node.name]
    children(net, index)
end

function discrete_ancestors(net::EnhancedBayesianNetwork, node::AbstractNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents(net, node)[3])
    continuous_parents = filter(x -> isa(x, ContinuousNode), parents(net, node)[3])
    if isempty(continuous_parents)
        return discrete_parents
    end
    return unique([discrete_parents..., mapreduce(x -> discrete_ancestors(net, x), vcat, continuous_parents)
    ...])
end

function _verify_net(net::AbstractNetwork)
    map(n -> _verify_child_node(net, n), net.nodes)
    functional_nodes = filter(x -> isa(x, FunctionalNode), net.nodes)
    map(fn -> _verify_functional_node(net, fn), functional_nodes)
    return nothing
end

function _verify_functional_node(net::EnhancedBayesianNetwork, node::FunctionalNode)
    pars = parents(net, node)[3]
    if isempty(pars)
        error("functional node '$(node.name)' must have at least one parent")
    end
    cont_pars = filter(x -> isa(x, ContinuousNode), pars)
    if isempty(cont_pars)
        @warn "functional node '$(node.name)' have no continuous parents. All the simulations will return the same output"
    end
    disc_pars = filter(x -> isa(x, DiscreteNode), pars)
    map(dp -> _non_empty_parameters_vector(net, dp), disc_pars)
    return nothing
end

function _verify_child_node(net::EnhancedBayesianNetwork, node::AbstractNode)
    if !_is_root(node) && !isa(node, FunctionalNode)
        th_parents_names = Symbol.(names(node.cpt[!, Not(node.name, :Prob)]))
        if !issetequal(th_parents_names, parents(net, node)[2])
            error("node '$(node.name)''s cpt requires exctly the nodes '$th_parents_names' to be its parents, but provided parents are '$(parents(net, node)[2])'")
        end
        th_scenarios = _theoretical_scenarios(net, node)
        cpt_scenarios = _scenarios(node)
        if cpt_scenarios != th_scenarios
            error("node '$(node.name)' has defined cpt scenarios $(node.cpt) not coherent with the theoretical one $th_scenarios")
        end
    end
    return nothing
end

function _non_empty_parameters_vector(net::EnhancedBayesianNetwork, node::DiscreteNode)
    chs = children(net, node)[3]
    if any(isa.(chs, FunctionalNode)) && isempty(node.parameters)
        error("node '$(node.name)' is a discrete parent of a functional node and cannot have an empty parameters vector")
    end
    return nothing
end

function _theoretical_scenarios(net::EnhancedBayesianNetwork, node::AbstractNode)
    par = discrete_ancestors(net, node)
    discrete_parents = filter(x -> isa(x, DiscreteNode), par)
    function f(par)
        return map(st -> (par.name => st), EnhancedBayesianNetworks._states(par))
    end
    discrete_parents_combination = Iterators.product(f.(discrete_parents)...)
    discrete_parents_combination = map(t -> [t...], discrete_parents_combination)
    return Dict.(vec(discrete_parents_combination))
end

function _remove_node!(net::AbstractNetwork, index::Int64)
    adj_matrix = net.adj_matrix[1:end.!=index, 1:end.!=index]
    nodes = deleteat!(net.nodes, index)
    topology_vec = collect(net.topology_dict)
    function f(kv, i)
        if kv[2] > i
            return Pair(kv[1], kv[2] - 1)
        elseif kv[2] != i
            return kv
        end
    end
    topology_vec = map(t -> f(t, index), topology_vec)
    filter!(x -> !isnothing(x), topology_vec)
    topology_dict = Dict(topology_vec)
    net.adj_matrix = adj_matrix
    net.topology_dict = topology_dict
    net.nodes = nodes
    return nothing
end

function _remove_node!(net::AbstractNetwork, name::Symbol)
    index = net.topology_dict[name]
    _remove_node!(net, index)
end

function _remove_node!(net::AbstractNetwork, node::AbstractNode)
    index = net.topology_dict[node.name]
    _remove_node!(net, index)
end

function _add_node!(net::AbstractNetwork, node::AbstractNode)
    push!(net.nodes, node)
    net.topology_dict[node.name] = length(net.nodes)
    net.adj_matrix = hcat(net.adj_matrix, zeros(net.adj_matrix.m))
    net.adj_matrix = vcat(net.adj_matrix, zeros(net.adj_matrix.n)')
    return nothing
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

function markov_envelope(net::EnhancedBayesianNetwork)
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