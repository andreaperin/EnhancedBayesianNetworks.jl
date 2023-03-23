using Plots
using GraphRecipes
using Graphs: DiGraph, SimpleEdge, add_edge!, rem_edge!,
    add_vertex!, rem_vertex!,
    edges, topological_sort_by_dfs, inneighbors,
    outneighbors, is_cyclic, nv, ne,
    outdegree, bfs_tree, dst

include("nodes.jl")

abstract type ProbabilisticGraphicalModel end
abstract type AbstractBayesNet <: ProbabilisticGraphicalModel end


# """
# A Standard Bayes Network Struct to be used when there are no Functional Nodes.
# """
# mutable struct StdBayesNet <: AbstractBayesNet
#     dag::DiGraph
#     nodes::Vector{T} where {T<:AbstractNode}
#     cpds::Vector{CPD}
#     name_to_index::Dict{NodeName,Int}
#     ## Check none of the nodes is FunctionalNode
#     function StdBayesNet(dag::DiGraph, nodes::Vector{T}, cpds::Vector{CPD}, name_to_index::Dict{NodeName,Int}) where {T<:AbstractNode}
#         if isa.(nodes, FunctionalNode) == zeros(length(nodes))
#             new(dag, nodes, cpds, name_to_index)
#         else
#             nodes_names = name.(nodes[isa.(nodes, FunctionalNode)])
#             throw(DomainError(nodes_names, "StdBayesNet cannot handle Functional Nodes => Pass to EnhancedBayesNet"))
#         end
#     end
# end

# function StdBayesNet(nodes::Vector{F}) where {F<:AbstractNode}
#     dag = _build_DiAGraph(nodes)
#     ## Check Graph's a-cyclicity
#     !is_cyclic(dag) || throw(DomainError(dag, "BayesNet graph is non-acyclic!"))
#     ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
#     return StdBayesNet(ordered_dag, ordered_nodes, ordered_cpds, ordered_name_to_index)
# end

# Base.get(bn::StdBayesNet, i::Int) = bn.cpds[i]
# Base.get(bn::StdBayesNet, nodename::NodeName) = bn.cpds[bn.name_to_index[nodename]]
# Base.length(bn::StdBayesNet) = length(bn.cpds)

"""
An Enhanced Bayes Network Struct to be used when there is at least one Functional Nodes.
"""
mutable struct EnhancedBayesNet <: AbstractBayesNet
    dag::DiGraph
    nodes::Vector{T} where {T<:AbstractNode}
    cpds::Vector{CPD}
    name_to_index::Dict{NodeName,Int}
end


function EnhancedBayesNet(nodes::Vector{F}) where {F<:AbstractNode}
    dag = _build_DiAGraph(nodes)
    ## Check Graph's a-cyclicity
    !is_cyclic(dag) || throw(DomainError(dag, "BayesNet graph is non-acyclic!"))
    ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
    return EnhancedBayesNet(ordered_dag, ordered_nodes, ordered_cpds, ordered_name_to_index)
end


Base.get(ebn::EnhancedBayesNet, i::Int) = ebn.cpds[i]
Base.get(ebn::EnhancedBayesNet, nodename::NodeName) = ebn.cpds[ebn.name_to_index[nodename]]
Base.length(ebn::EnhancedBayesNet) = length(ebn.cpds)
index_to_name(ebn::EnhancedBayesNet) = Dict(value => key for (key, value) in ebn.name_to_index)


## Functions for build BayesNet struct

function _build_DiAGraph(nodes::Vector{F}) where {F<:AbstractNode}
    cpds = [i.cpd for i in nodes]
    name_to_index = Dict{NodeName,Int}()
    for (i, cpd) in enumerate(cpds)
        name_to_index[name(cpd)] = i
    end
    dag = DiGraph(length(cpds))
    for cpd in cpds
        j = name_to_index[name(cpd)]
        for p in parents(cpd)
            i = name_to_index[p]
            add_edge!(dag, i, j)
        end
    end
    return dag
end

function _build_DiAGraph(cpds::Vector{CPD})
    name_to_index = Dict{NodeName,Int}()
    for (i, cpd) in enumerate(cpds)
        name_to_index[name(cpd)] = i
    end
    dag = DiGraph(length(cpds))
    for cpd in cpds
        j = name_to_index[name(cpd)]
        for p in parents(cpd)
            i = name_to_index[p]
            add_edge!(dag, i, j)
        end
    end
    return dag
end

function _topological_ordered_dag(nodes::Vector{F}) where {F<:AbstractNode}
    dag = _build_DiAGraph(nodes)
    topological_ordered_vect = topological_sort_by_dfs(dag)
    cpds = [i.cpd for i in nodes]
    new_cpds = Vector{CPD}(undef, length(cpds))
    new_nodes = Vector{AbstractNode}(undef, length(nodes))
    new_name_to_index = Dict{NodeName,Int}()
    for (new_index, old_index) in enumerate(topological_ordered_vect)
        new_name_to_index[name(cpds[old_index])] = new_index
        new_cpds[new_index] = cpds[old_index]
        new_nodes[new_index] = nodes[old_index]
    end
    new_dag = _build_DiAGraph(new_cpds)
    return new_cpds, new_nodes, new_name_to_index, new_dag
end


"""
Utilisties Function
"""
function show(bn::AbstractBayesNet)
    graphplot(
        bn.dag,
        method=:tree,
        names=name.(bn.nodes),
        fontsize=9,
        nodeshape=:ellipse,
        markercolor=map(x -> x.type == "discrete" ? "lightgreen" : "orange", bn.nodes),
        linecolor=:darkgrey,
    )
end

function node_from_nodename(ebn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    return filter(x -> name(x) == nodename, ebn.nodes)[1]
end


function _topological_ordered_bn!(bn::M) where {M<:AbstractBayesNet}
    new_cpds, new_nodes, new_name_to_index, new_dag = _topological_ordered_dag(bn.nodes)
    bn.dag = new_dag
    bn.cpds = new_cpds
    bn.nodes = new_nodes
    bn.name_to_index = new_name_to_index
    bn
end

"""
Returns the ordered list of NodeNames
"""
function Base.names(ebn::M) where {M<:AbstractBayesNet}
    retval = Array{NodeName}(undef, length(ebn))
    for (i, cpd) in enumerate(ebn.cpds)
        retval[i] = name(cpd)
    end
    retval
end

"""
Returns the parents as a list of NodeNames
"""
parents(ebn::M, nodename::NodeName) where {M<:AbstractBayesNet} = parents(get(ebn, nodename))
"""
Returns the children as a list of NodeNames
"""
function children(ebn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    i = ebn.name_to_index[nodename]
    NodeName[name(ebn.cpds[j]) for j in outneighbors(ebn.dag, i)]
end

"""
Returns all neighbors as a list of NodeNames.
"""
function neighbors(ebn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    i = ebn.name_to_index[nodename]
    NodeName[name(ebn.cpds[j]) for j in append!(inneighbors(ebn.dag, i), outneighbors(ebn.dag, i))]
end

"""
Returns all descendants as a list of NodeNames.
"""
# dst(edge::Pair{Int,Int}) = edge[2] # Graphs used to return a Pair, now it returns a SimpleEdge
function descendants(ebn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    retval = Set{Int}()
    for edge in edges(bfs_tree(ebn.dag, ebn.name_to_index[nodename]))
        push!(retval, edge.dst)
    end
    NodeName[name(ebn.cpds[i]) for i in sort!(collect(retval))]
end

"""
Return the children, parents, and parents of children (excluding target) as a Set of NodeNames
"""
function markov_blanket(ebn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    nodeNames = NodeName[]
    for child in children(ebn, nodename)
        append!(nodeNames, parents(ebn, child))
        push!(nodeNames, child)
    end
    append!(nodeNames, parents(ebn, nodename))
    return setdiff(Set(nodeNames), Set(NodeName[nodename]))
end

"""
Whether the BayesNet contains the given edge
"""
function has_edge(ebn::M, parent::NodeName, child::NodeName)::Bool where {M<:AbstractBayesNet}
    u = get(ebn.name_to_index, parent, 0)
    v = get(ebn.name_to_index, child, 0)
    u != 0 && v != 0 && Graphs.has_edge(ebn.dag, u, v)
end

"""
Returns whether the set of node names `x` is d-separated from the set `y` given the set `given`
"""
function is_independent(ebn::M, x::NodeNames, y::NodeNames, given::NodeNames) where {M<:AbstractBayesNet}
    start_node = x[1]
    finish_node = y[1]
    if start_node == finish_node
        return true
    end
    C = Set(given)
    analyzed_nodes = Set()
    # Find all paths from x to y
    paths = []
    path_queue = []
    push!(analyzed_nodes, start_node)
    for next_node in neighbors(ebn, start_node)
        if !in(next_node, analyzed_nodes)
            push!(analyzed_nodes, next_node)
            push!(path_queue, [start_node, next_node])
        end
    end
    while (!isempty(path_queue))
        cur_path = pop!(path_queue)
        last_node = cur_path[end]
        for next_node in neighbors(ebn, last_node)
            if next_node == finish_node
                push!(paths, push!(copy(cur_path), next_node))
            elseif !in(next_node, analyzed_nodes)
                push!(analyzed_nodes, next_node)
                push!(path_queue, push!(copy(cur_path), next_node))
            end
        end
    end
    # Check each path to see if it contains information indicating d-separation
    for path in paths
        is_d_separated = false
        if length(path) == 2
            is_d_separated = true
        else
            # Examine all middle nodes
            for i in 2:(length(path)-1)
                prev_node = path[i-1]
                cur_node = path[i]
                next_node = path[i+1]
                # Check for chain or fork (first or second d-separation criteria)
                if in(cur_node, C)
                    # Chain
                    if in(cur_node, children(ebn, prev_node)) && in(next_node, children(ebn, cur_node))
                        is_d_separated = true
                        break
                        # Fork
                    elseif in(prev_node, children(ebn, cur_node)) && in(next_node, children(ebn, cur_node))
                        is_d_separated = true
                        break
                    end
                    # Check for v-structure (third d-separation criteria)
                else
                    if in(cur_node, children(ebn, prev_node)) && in(cur_node, children(ebn, next_node))
                        descendant_list = descendants(ebn, cur_node)
                        descendants_in_C = false
                        for d in descendant_list
                            if in(d, C)
                                descendants_in_C = true
                                break
                            end
                        end
                        if !descendants_in_C
                            is_d_separated = true
                            break
                        end
                    end
                end
            end
        end
        if !is_d_separated
            return false
        end
    end
    # All paths are d-separated, so x and y are conditionally independent.
    return true
end

function _get_continous_nodenames_in_markov_blanket(ebn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    blanket = collect(markov_blanket(ebn, nodename))
    blanket_nodes = node_from_nodename.(repeat([ebn], length(blanket)), blanket)
    blanket_continuous_nodenames = name.(filter(x -> x.type == "continuous", blanket_nodes))
    return blanket_continuous_nodenames
end

function markov_envelopes(ebn::M) where {M<:AbstractBayesNet}
    continuous_nodenames = name.(filter(x -> x.type == "continuous", ebn.nodes))
    groups = []
    for continuous_nodename in continuous_nodenames
        blanket_continuous_nodenames = _get_continous_nodenames_in_markov_blanket(ebn, continuous_nodename)
        group = vcat(continuous_nodename, blanket_continuous_nodenames)
        while ~isempty(blanket_continuous_nodenames)
            blanket_i = _get_continous_nodenames_in_markov_blanket(ebn, blanket_continuous_nodenames[1])
            popfirst!(blanket_continuous_nodenames)
            vcat(blanket_continuous_nodenames, collect(setdiff(blanket_i, blanket_continuous_nodenames)))
            vcat(group, collect(setdiff(blanket_i, group)))
        end
        isempty(setdiff(group, continuous_nodenames)) ? groups = [group] : push!(groups, group)
    end
    envelopes = []
    for group in unique(sort.(groups))
        all_blankets = markov_blanket.(repeat([ebn], length(group)), group)
        push!(envelopes, unique(Iterators.flatten(all_blankets)))
    end
    return envelopes
end

## TODO add function revert single node based on DiGraph (adjacent forward node fadj or badj)
function _invert_link(dag::SimpleDiGraph, parent_index::Int64, child_index::Int64)
    new_dag = copy(dag)
    child_index ∉ ebn.dag.fadjlist[parent_index] && throw(DomainError("$parent_index is not a parent of $child_index"))
    rem_edge!(new_dag, parent_index, child_index)
    add_edge!(new_dag, child_index, parent_index)
    is_cyclic(new_dag) ? throw(DomainError(new_dag, "BayesNet graph is non-acyclic!")) : return new_dag
end

function _invert_nodes_link(dag::SimpleDiGraph, parent_index::Int64, child_index::Int64)
    parents_pr = dag.badjlist[parent_index]
    parents_ch = setdiff(dag.badjlist[child_index], parent_index)
    new_dag = _invert_link(dag, parent_index, child_index)
    [add_edge!(new_dag, i, parent_index) for i in parents_ch]
    [add_edge!(new_dag, j, child_index) for j in parents_pr]
    is_cyclic(new_dag) ? throw(DomainError("reverting $parent_index and $child_index a cyclic dag is created")) : new_dag = new_dag
    return new_dag
end

# function _invert_nodes_link(dag::SimpleDiGraph, parent_index::Int64, child_index::Int64)
#     new_dag = _invert_link(dag, parent_index, child_index)
#     parents_pr = dag.badjlist[parent_index]
#     parents_ch = setdiff(dag.badjlist[child_index], parent_index)
#     [add_edge!(new_dag, i, parent_index) for i in parents_ch]
#     [add_edge!(new_dag, j, child_index) for j in parents_pr]
#     is_cyclic(new_dag) ? throw(DomainError("reverting $parent_index and $child_index a cyclic dag is created")) : return new_dag
# end

function _is_barren_node(dag::SimpleDiGraph, node_index::Int64)
    return isempty(dag.fadjlist[node_index])
end

function _remove_barren_nodes(dag::SimpleDiGraph, barren_index::Int64)
    for i in copy(dag.badjlist[barren_index])
        rem_edge!(dag, i, barren_index)
    end
    deleteat!(dag.badjlist, barren_index)
    new_badjlist = Vector{Vector{Int64}}()
    for i in dag.badjlist
        push!(new_badjlist, i .- Int.(i .> barren_index))
    end
    deleteat!(dag.fadjlist, barren_index)
    new_fadjlist = Vector{Vector{Int64}}()
    for i in dag.fadjlist
        push!(new_fadjlist, i .- Int.(i .> barren_index))
    end
    return SimpleDiGraph(5, new_fadjlist, new_badjlist)
end

function _eliminate_node(ebn::M, node::NodeName) where {M<:AbstractBayesNet}
    node_index = ebn.name_to_index[node]
    child_nodes = children(ebn, node)
    new_dag = ebn.dag
    for child in child_nodes
        new_dag = _invert_nodes_link(new_dag, ebn.name_to_index[node], ebn.name_to_index[child])
    end
    _is_barren_node(new_dag, node_index) ? new_dag = _remove_barren_nodes(new_dag, node_index) : new_dag = new_dag
    return new_dag, deleteat!(name.(ebn.nodes), node_index)
end
