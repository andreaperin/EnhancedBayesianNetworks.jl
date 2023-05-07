struct ReducedBayesianNetwork <: ProbabilisticGraphicalModel
    dag::SimpleDiGraph
    nodes::Vector{<:AbstractNode}
    name_to_index::Dict{Symbol,Int}
end

function get_children(ebn::ReducedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in outneighbors(ebn.dag, i)]
end

function get_parents(ebn::ReducedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in inneighbors(ebn.dag, i)]
end

function get_neighbors(ebn::ReducedBayesianNetwork, node::N) where {N<:AbstractNode}
    i = ebn.name_to_index[node.name]
    [ebn.nodes[j] for j in append!(inneighbors(ebn.dag, i), outneighbors(ebn.dag, i))]
end

function plot(ebn::ReducedBayesianNetwork)
    graphplot(
        ebn.dag,
        names=[i.name for i in ebn.nodes],
        # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), ebn.nodes),
        font_size=10,
        node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, ebn.nodes),
        markercolor=map(x -> isa(x, DiscreteFunctionalNode) ? "lightgreen" : "orange", ebn.nodes),
        linecolor=:darkgrey,
    )
end

function reduce_ebn_markov_envelopes(ebn::EnhancedBayesianNetwork)
    markov_envelopes = markov_envelope(ebn)
    indipendent_ebns = _create_ebn_from_envelope.(repeat([ebn], length(markov_envelopes)), markov_envelopes)
    reduce_ebn_standard.(indipendent_ebns)
end

function reduce_ebn_standard(ebn::EnhancedBayesianNetwork)
    ## Always starts with the link to the continuous nodes with fewest parents
    continuous_nodes = filter(x -> isa(x, ContinuousNode), ebn.nodes)
    r_dag_nodes = copy(ebn.nodes)
    r_dag = copy(ebn.dag)
    while !isempty(continuous_nodes)
        starting_node = continuous_nodes[findmin(map(x -> length(get_parents(ebn, x)), continuous_nodes))[2]]
        starting_node_index = findfirst(x -> x == starting_node, r_dag_nodes)
        r_dag = _reduce_continuousnode(r_dag, starting_node_index)

        r_dag_nodes = deleteat!(r_dag_nodes, starting_node_index)
        deleteat!(continuous_nodes, findall(x -> x == starting_node, continuous_nodes))
    end
    r_name_to_index = Dict(x.name => i for (i, x) in enumerate(r_dag_nodes))
    return ReducedBayesianNetwork(r_dag, r_dag_nodes, r_name_to_index)
end


```
Dag Operations
```
function _reduce_continuousnode(dag::SimpleDiGraph, node_index::Int)
    r_dag = copy(dag)
    child_indices = r_dag.fadjlist[node_index]
    for child in child_indices
        r_dag = _invert_link_nodes(r_dag, node_index, child)
    end
    _remove_barren_node(r_dag, node_index)
end


function _invert_link_dag(dag::SimpleDiGraph, parent_index::Int, child_index::Int)
    new_dag = copy(dag)
    child_index ∉ dag.fadjlist[parent_index] && error("Invalid dag-link to be inverted")
    rem_edge!(new_dag, parent_index, child_index)
    add_edge!(new_dag, child_index, parent_index)
    is_cyclic(new_dag) ? error("Cyclic dag error") : return new_dag
end

function _invert_link_nodes(dag::SimpleDiGraph, parent_index::Int, child_index::Int)
    parents_pr = dag.badjlist[parent_index]
    parents_ch = setdiff(dag.badjlist[child_index], parent_index)
    new_dag = _invert_link_dag(dag, parent_index, child_index)
    [add_edge!(new_dag, i, parent_index) for i in parents_ch]
    [add_edge!(new_dag, j, child_index) for j in parents_pr]
    is_cyclic(new_dag) ? error("Cyclic dag error") : new_dag = new_dag
    return new_dag
end

function _remove_barren_node(dag::SimpleDiGraph, node_index::Int)
    !isempty(dag.fadjlist[node_index]) && error("node to be eliminated must be a barren node")
    for i in copy(dag.badjlist[node_index])
        rem_edge!(dag, i, node_index)
    end
    deleteat!(dag.badjlist, node_index)
    new_badjlist = Vector{Vector{Int64}}()
    for i in dag.badjlist
        push!(new_badjlist, i .- Int.(i .> node_index))
    end
    deleteat!(dag.fadjlist, node_index)
    new_fadjlist = Vector{Vector{Int64}}()
    for i in dag.fadjlist
        push!(new_fadjlist, i .- Int.(i .> node_index))
    end
    return SimpleDiGraph(dag.ne, new_fadjlist, new_badjlist)
end

##TODO test
function _get_node_given_state(rbn::ReducedBayesianNetwork, state::Symbol)
    nodes = filter(x -> !isa(x, DiscreteFunctionalNode) && isa(x, DiscreteNode), rbn.nodes)
    [node for node in nodes if state ∈ _get_states(node)][1]
end

```
Reduced BN Operations
```
function _build_structuralreliabilityproblem_node(rbn::ReducedBayesianNetwork, node::DiscreteFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), get_parents(rbn, node))
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    discrete_parents_combination = vec(collect(Iterators.product(_get_states.(discrete_parents)...)))
    discrete_parents_combination = map(x -> [i for i in x], discrete_parents_combination)
    srps = OrderedDict{Vector{Symbol},StructuralReliabilityProblem}()
    for combination in discrete_parents_combination
        evidence = [(s, _get_node_given_state(rbn, s)) for s in combination]

        uq_parameters = vcat([get_parameters(p, evidence)[1] for p in discrete_parents]...)
        uq_randomvariables = vcat([get_randomvariable(p, evidence) for p in continuous_parents])
        uqinputs = vcat(uq_parameters, uq_randomvariables)

        models = get_models(node, evidence)

        srps[combination] = StructuralReliabilityProblem(models, uqinputs)
    end
    return StructuralReliabilityProblemNode(node.name, node.parents, srps)
end

function _get_failure_probability(node::StructuralReliabilityProblemNode)
    # for (comb, srp) in node.srps
    comb = [:yes, :n]
    srp = node.srps[comb]
    UncertaintyQuantification.sample(srp.inputs, 1000)





end