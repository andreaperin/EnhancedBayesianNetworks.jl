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
    [ebn.nodes[j] for j in unique(append!(inneighbors(ebn.dag, i), outneighbors(ebn.dag, i)))]
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
    rbn = deepcopy(ebn)
    continuous_nodes = filter(x -> isa(x, ContinuousNode), rbn.nodes)
    r_dag_nodes = copy(rbn.nodes)
    r_dag = rbn.dag
    while !isempty(continuous_nodes)
        starting_node = continuous_nodes[findmin(map(x -> length(get_parents(rbn, x)), continuous_nodes))[2]]
        starting_node_index = findall(isequal.(repeat([starting_node], length(r_dag_nodes)), r_dag_nodes))[1]

        children = get_children(rbn, starting_node)

        r_dag = _reduce_continuousnode(r_dag, starting_node_index)

        for child in children
            for parent in child.parents
                isequal(parent, starting_node) && deleteat!(child.parents, findall(x -> x == parent, child.parents))
            end
            if !isa(starting_node, ContinuousRootNode)
                starting_node_parents = filter(x -> isa(x, DiscreteNode), starting_node.parents)
                for s in starting_node_parents
                    .!any(isequal.(repeat([s], length(child.parents)), child.parents)) && push!(child.parents, s)
                end
            end
        end

        r_dag_nodes = deleteat!(r_dag_nodes, starting_node_index)
        deleteat!(continuous_nodes, findall(x -> x == starting_node, continuous_nodes))
    end

    ordered_rdag, ordered_rnodes, ordered_rname_to_index = _topological_ordered_dag(r_dag_nodes)

    return ReducedBayesianNetwork(ordered_rdag, ordered_rnodes, ordered_rname_to_index)
end


```
Dag Operations
```
function _reduce_continuousnode(dag::SimpleDiGraph, node_index::Int)
    r_dag = deepcopy(dag)
    child_indices = r_dag.fadjlist[node_index]
    for child in child_indices
        r_dag = _invert_link_nodes(r_dag, node_index, child)
    end
    _remove_barren_node(r_dag, node_index)
end


function _invert_link_dag(dag::SimpleDiGraph, parent_index::Int, child_index::Int)
    new_dag = deepcopy(dag)
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
    for i in deepcopy(dag.badjlist[node_index])
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

function _get_node_given_state(rbn::ReducedBayesianNetwork, state::Symbol)
    nodes = filter(x -> !isa(x, DiscreteFunctionalNode) && isa(x, DiscreteNode), rbn.nodes)
    [node for node in nodes if state ∈ _get_states(node)][1]
end

```
Reduced BN Operations
```
function evaluate_ebn(ebn::EnhancedBayesianNetwork, markov::Bool=true)
    markov ? rbns = reduce_ebn_markov_envelopes(ebn) : rbns = [reduce_ebn_standard(ebn)]
    for rbn in rbns
        functional_nodes = filter(x -> isa(x, DiscreteFunctionalNode), rbn.nodes)
        while !isempty(functional_nodes)
            node = functional_nodes[1]
            if isempty(filter(x -> isa(x, DiscreteFunctionalNode), node.parents))
                srp_node = _build_structuralreliabilityproblem_node(rbn, ebn, node)
                node.parents = srp_node.parents
                node.pf, node.cov, node.samples = _get_failure_probability(srp_node)
                popfirst!(functional_nodes)
            else
                push!(functional_nodes, node)
                popfirst!(functional_nodes)
            end
        end
    end
    return rbns
end

function _build_structuralreliabilityproblem_node(rbn::ReducedBayesianNetwork, ebn::EnhancedBayesianNetwork, node::DiscreteFunctionalNode)
    ebn_discrete_parents = filter(x -> isa(x, DiscreteNode), get_parents(ebn, node))
    ebn_continuous_parents = filter(x -> isa(x, ContinuousNode), get_parents(ebn, node))

    rbn_discrete_parents = filter(x -> isa(x, DiscreteNode), get_parents(rbn, node))
    rbn_discrete_parents_combination = vec(collect(Iterators.product(_get_states.(rbn_discrete_parents)...)))
    rbn_discrete_parents_combination = map(x -> [i for i in x], rbn_discrete_parents_combination)
    srps = OrderedDict{Vector{Symbol},StructuralReliabilityProblem}()

    for evidence in rbn_discrete_parents_combination

        uq_parameters = mapreduce(p -> get_parameters(p, evidence), vcat, ebn_discrete_parents)
        uq_randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, ebn_continuous_parents)
        uqinputs = vcat(uq_parameters, uq_randomvariables)

        ebn_node = filter(x -> x.name == node.name, ebn.nodes)[1]
        ordered_functional_node = [ebn_node]
        get_cont_fun_parents = n -> filter(x -> isa(x, ContinuousFunctionalNode), n.parents)

        cont_fun_parents = get_cont_fun_parents(ebn_node)
        while !isempty(cont_fun_parents)
            append!(ordered_functional_node, cont_fun_parents)
            cont_fun_parents = mapreduce(p -> get_cont_fun_parents(p), vcat, cont_fun_parents)
        end
        get_models(ordered_functional_node[1], evidence)
        models = mapreduce(p -> get_models(p, evidence), vcat, reverse(ordered_functional_node))

        performances = get_performance(ebn_node, evidence)

        simulations = get_simulation(ebn_node, evidence)

        srps[evidence] = StructuralReliabilityProblem(models, uqinputs, performances, simulations)
    end
    e = collect(keys(srps))[1]
    parents = _get_node_given_state.(repeat([rbn], length(e)), e)
    return StructuralReliabilityProblemNode(node.name, parents, srps)
end

##TODO (incomplete and not reliable TEST)
function _get_failure_probability(node::StructuralReliabilityProblemNode)
    pf = Dict()
    cov = Dict()
    samples = Dict()
    for (comb, srp) in node.srps
        pf[comb] = probability_of_failure(srp.models, srp.performance, srp.inputs, srp.simulation)[1]
        cov[comb] = probability_of_failure(srp.models, srp.performance, srp.inputs, srp.simulation)[2]
        samples[comb] = probability_of_failure(srp.models, srp.performance, srp.inputs, srp.simulation)[3]
    end
    return pf, cov, samples
end


# Base.get(rbn::ReducedBayesianNetwork, node::Symbol)
