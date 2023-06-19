struct ConditionalProbabilityDistribution
    target::Symbol
    parents::Vector{Symbol}
    parents_states_mapping_dict::Dict{Symbol,Dict{Symbol,Int}}
    parental_ncategories::Vector{Int}
    states::Vector{Symbol}
    distributions::Dict{Vector{Symbol},Dict{Symbol,Real}}
end
mutable struct BayesianNetwork <: ProbabilisticGraphicalModel
    dag::SimpleDiGraph
    nodes::Vector{<:DiscreteNode}
    name_to_index::Dict{Symbol,Int}

    function BayesianNetwork(dag::DiGraph, nodes::Vector{AbstractNode}, name_to_index::Dict{Symbol,Int})
        any([!isa(x, DiscreteNode) for x in nodes]) && error("Bayesian Network allows discrete node only!")
        nodes = Vector{DiscreteNode}(nodes)
        new(dag, nodes, name_to_index)
    end
end

function BayesianNetwork(nodes::Vector{<:AbstractNode})
    ordered_dag, ordered_nodes, ordered_name_to_index = _topological_ordered_dag(nodes)
    BayesianNetwork(ordered_dag, ordered_nodes, ordered_name_to_index)
end

function BayesianNetwork(rbn::ReducedBayesianNetwork)
    functional_nodes = filter(x -> isa(x, DiscreteFunctionalNode), rbn.nodes)
    any(isempty.([i.pf for i in functional_nodes])) && error("rbn needs to evaluated!")
    bn_nodes = filter(x -> !isa(x, DiscreteFunctionalNode), rbn.nodes)
    append!(bn_nodes, _get_discretestandardnode.(functional_nodes))
    BayesianNetwork(bn_nodes)
end


function get_cpd(bn::BayesianNetwork, i::Int)
    n = bn.nodes[i]
    target = n.name
    st = _get_states(bn.nodes[i])
    isa(n, RootNode) ? parents = Vector{Symbol}() : parents = [x.name for x in n.parents]
    isa(n, RootNode) ? parental_ncategories = Vector{Int}() : parental_ncategories = map(s -> length(_get_states(s)), n.parents)
    isa(n, RootNode) ? distributions = Dict(Vector{Symbol}() => n.states) : distributions = n.states
    parents_nodes = [bn.nodes[bn.name_to_index[s]] for s in parents]
    mapping_dict = Dict{Symbol,Dict{Symbol,Int}}()
    for node in parents_nodes
        mapping_dict[node.name] = Dict(s => i for (i, s) in enumerate(_get_states(node)))
    end

    ConditionalProbabilityDistribution(target, parents, mapping_dict, parental_ncategories, st, distributions)
end

get_cpd(bn::BayesianNetwork, name::Symbol) = get_cpd(bn, bn.name_to_index[name])

function plot(bn::BayesianNetwork)
    graphplot(
        bn.dag,
        names=[i.name for i in bn.nodes],
        # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), bn.nodes),
        font_size=10,
        node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, bn.nodes),
        markercolor=map(x -> isa(x, DiscreteFunctionalNode) ? "lightgreen" : "orange", bn.nodes),
        linecolor=:darkgrey,
    )
end

function _get_discretestandardnode(node::DiscreteFunctionalNode)
    states = Dict{Vector{Symbol},Dict{Symbol,Float64}}()
    for (k, v) in node.pf
        states[k] = Dict(:f => v, :s => 1 - v)
    end
    return DiscreteStandardNode(node.name, node.parents, states)
end