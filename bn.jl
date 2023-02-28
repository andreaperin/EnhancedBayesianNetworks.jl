using Plots
using GraphRecipes
using Graphs: DiGraph, SimpleEdge, add_edge!, rem_edge!,
    add_vertex!, rem_vertex!,
    edges, topological_sort_by_dfs, inneighbors,
    outneighbors, is_cyclic, nv, ne,
    outdegree, bfs_tree, dst

include("CPDs.jl")
include("nodes.jl")

abstract type ProbabilisticGraphicalModel end
abstract type AbstractBayesNet <: ProbabilisticGraphicalModel end


"""
A Standard Bayes Network Struct to be used when there are no Functional Nodes.
"""
mutable struct StdBayesNet <: AbstractBayesNet
    dag::DiGraph
    nodes::Vector{T} where {T<:AbstractNode}
    cpds::Vector{CPD}
    name_to_index::Dict{NodeName,Int}
    ## Check none of the nodes is FunctionalNode
    function StdBayesNet(dag::DiGraph, nodes::Vector{T}, cpds::Vector{CPD}, name_to_index::Dict{NodeName,Int}) where {T<:AbstractNode}
        if isa.(nodes, FunctionalNode) == zeros(length(nodes))
            new(dag, nodes, cpds, name_to_index)
        else
            nodes_names = name.(nodes[isa.(nodes, FunctionalNode)])
            throw(DomainError(nodes_names, "StdBayesNet cannot handle Functional Nodes => Pass to EnhancedBayesNet"))
        end
    end
end

function StdBayesNet(nodes::Vector{F}) where {F<:AbstractNode}
    dag = _build_DiAGraph_from_nodes(nodes)
    ## Check Graph's a-cyclicity
    !is_cyclic(dag) || throw(DomainError(dag, "BayesNet graph is non-acyclic!"))
    ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
    return StdBayesNet(ordered_dag, ordered_nodes, ordered_cpds, ordered_name_to_index)
end

Base.get(bn::StdBayesNet, i::Int) = bn.cpds[i]
Base.get(bn::StdBayesNet, nodename::NodeName) = bn.cpds[bn.name_to_index[nodename]]
Base.length(bn::StdBayesNet) = length(bn.cpds)

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
    dag = _build_DiAGraph_from_nodes(nodes)
    ## Check Graph's a-cyclicity
    !is_cyclic(dag) || throw(DomainError(dag, "BayesNet graph is non-acyclic!"))
    ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
    return EnhancedBayesNet(ordered_dag, ordered_nodes, ordered_cpds, ordered_name_to_index)
end


function StdBayesNet(nodes::Vector{F}) where {F<:AbstractNode}
    dag = _build_DiAGraph_from_nodes(nodes)
    ## Check Graph's a-cyclicity
    !is_cyclic(dag) || throw(DomainError(dag, "BayesNet graph is non-acyclic!"))
    ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
    return StdBayesNet(ordered_dag, ordered_nodes, ordered_cpds, ordered_name_to_index)
end

Base.get(bn::EnhancedBayesNet, i::Int) = bn.cpds[i]
Base.get(bn::EnhancedBayesNet, nodename::NodeName) = bn.cpds[bn.name_to_index[nodename]]
Base.length(bn::EnhancedBayesNet) = length(bn.cpds)


## Functions for build BayesNet struct

function _build_DiAGraph_from_nodes(nodes::Vector{F}) where {F<:AbstractNode}
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

function _build_DiAGraph_from_nodes(
    cpds::Vector{CPD}
)
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
    dag = _build_DiAGraph_from_nodes(nodes)
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
    new_dag = _build_DiAGraph_from_nodes(new_cpds)
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
function Base.names(bn::M) where {M<:AbstractBayesNet}
    retval = Array{NodeName}(undef, length(bn))
    for (i, cpd) in enumerate(bn.cpds)
        retval[i] = name(cpd)
    end
    retval
end

"""
Returns the parents as a list of NodeNames
"""
parents(bn::M, nodename::NodeName) where {M<:AbstractBayesNet} = parents(get(bn, nodename))
"""
Returns the children as a list of NodeNames
"""
function children(bn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    i = bn.name_to_index[nodename]
    NodeName[name(bn.cpds[j]) for j in outneighbors(bn.dag, i)]
end

"""
Returns all neighbors as a list of NodeNames.
"""
function neighbors(bn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    i = bn.name_to_index[nodename]
    NodeName[name(bn.cpds[j]) for j in append!(inneighbors(bn.dag, i), outneighbors(bn.dag, i))]
end

"""
Returns all descendants as a list of NodeNames.
"""
# dst(edge::Pair{Int,Int}) = edge[2] # Graphs used to return a Pair, now it returns a SimpleEdge
function descendants(bn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    retval = Set{Int}()
    for edge in edges(bfs_tree(bn.dag, bn.name_to_index[nodename]))
        push!(retval, edge.dst)
    end
    NodeName[name(bn.cpds[i]) for i in sort!(collect(retval))]
end

"""
Return the children, parents, and parents of children (excluding target) as a Set of NodeNames
"""
function markov_blanket(bn::M, nodename::NodeName) where {M<:AbstractBayesNet}
    nodeNames = NodeName[]
    for child in children(bn, nodename)
        append!(nodeNames, parents(bn, child))
        push!(nodeNames, child)
    end
    append!(nodeNames, parents(bn, nodename))
    return setdiff(Set(nodeNames), Set(NodeName[nodename]))
end

"""
Whether the BayesNet contains the given edge
"""
function has_edge(bn::M, parent::NodeName, child::NodeName)::Bool where {M<:AbstractBayesNet}
    u = get(bn.name_to_index, parent, 0)
    v = get(bn.name_to_index, child, 0)
    u != 0 && v != 0 && Graphs.has_edge(bn.dag, u, v)
end

"""
Returns whether the set of node names `x` is d-separated from the set `y` given the set `given`
"""
function is_independent(bn::M, x::NodeNames, y::NodeNames, given::NodeNames) where {M<:AbstractBayesNet}
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
    for next_node in neighbors(bn, start_node)
        if !in(next_node, analyzed_nodes)
            push!(analyzed_nodes, next_node)
            push!(path_queue, [start_node, next_node])
        end
    end
    while (!isempty(path_queue))
        cur_path = pop!(path_queue)
        last_node = cur_path[end]
        for next_node in neighbors(bn, last_node)
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
                    if in(cur_node, children(bn, prev_node)) && in(next_node, children(bn, cur_node))
                        is_d_separated = true
                        break
                        # Fork
                    elseif in(prev_node, children(bn, cur_node)) && in(next_node, children(bn, cur_node))
                        is_d_separated = true
                        break
                    end
                    # Check for v-structure (third d-separation criteria)
                else
                    if in(cur_node, children(bn, prev_node)) && in(cur_node, children(bn, next_node))
                        descendant_list = descendants(bn, cur_node)
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