@auto_hash_equals mutable struct BayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{<:AbstractNode}
    topology_dict::Dict
    adj_matrix::SparseMatrixCSC

    function BayesianNetwork(nodes::AbstractVector{<:AbstractNode}, topology_dict::Dict, adj_matrix::SparseMatrixCSC)
        nodes_names = map(i -> i.name, nodes)
        if nodes_names != unique(nodes_names)
            error("network nodes names must be unique")
        end
        discrete_nodes = filter(x -> isa(x, DiscreteNode) && !isa(x, FunctionalNode), nodes)
        if !isempty(discrete_nodes)
            states_list = mapreduce(i -> _get_states(i), vcat, discrete_nodes)
            if states_list != unique(states_list)
                error("network nodes states must be unique")
            end
        end
        continuous_nodes = nodes[isa.(nodes, ContinuousNode)]
        continuous_nodes_names = [i.name for i in continuous_nodes]
        if !isempty(continuous_nodes)
            error("node/s $continuous_nodes_names are continuous. Use EnhancedBayesianNetwork structure!")
        end
        imprecise_nodes = nodes[_is_imprecise.(nodes)]
        imprecise_nodes_names = [i.name for i in imprecise_nodes]
        if !isempty(imprecise_nodes)
            error("node/s $imprecise_nodes_names are imprecise. Use CrealNetwork structure!")
        end
        new(nodes, topology_dict, adj_matrix)
    end
end

function BayesianNetwork(nodes::AbstractVector{<:AbstractNode})
    n = length(nodes)
    topology_dict = Dict()
    for (i, n) in enumerate(nodes)
        topology_dict[n.name] = i
    end
    adj_matrix = sparse(zeros(n, n))
    return BayesianNetwork(nodes, topology_dict, adj_matrix)
end

function BayesianNetwork(net::EnhancedBayesianNetwork)
    order!(net)
    nodes = net.nodes
    topology_dict = net.topology_dict
    adj_matrix = net.adj_matrix
    return BayesianNetwork(nodes, topology_dict, adj_matrix)
end

function get_cpd(bn::BayesianNetwork, i::Int)
    reverse_dict = Dict(value => key for (key, value) in bn.topology_dict)
    name = reverse_dict[i]
    node = first(filter(x -> x.name == name, bn.nodes))
    st = _get_states(node)

    _, parents_names, parents_nodes = get_parents(bn, i)

    parental_ncategories = map(n -> length(_get_states(n)), parents_nodes)
    if isa(node, RootNode)
        distribution = Dict(Vector{Symbol}() => node.states)
    else
        distribution = node.states
    end
    mapping_dict = Dict{Symbol,Dict{Symbol,Int}}()
    for node in parents_nodes
        mapping_dict[node.name] = Dict(s => i for (i, s) in enumerate(_get_states(node)))
    end
    ConditionalProbabilityDistribution(node.name, parents_names, mapping_dict, parental_ncategories, st, distribution)
end

get_cpd(bn::BayesianNetwork, name::Symbol) = get_cpd(bn, bn.topology_dict[name])
get_cpd(bn::BayesianNetwork, node::AbstractNode) = get_cpd(bn, node.name)