function _evaluate_single_layer(ebn::EnhancedBayesianNetwork)
    ## Discretization
    disc_ebn = discretize(ebn)
    ## Nodes to be evaluate from ebn
    functional_nodes = filter(x -> isa(x, FunctionalNode), disc_ebn.nodes)
    functional_nodes_to_eval = filter(x -> all(!isa(y, FunctionalNode) for y in x.parents), functional_nodes)
    res = map(n -> (n, _build_structuralreliabilityproblem_node(n)), functional_nodes_to_eval)
    ## rbn with StructuralReliabilityProblemNode
    srp_ebn = deepcopy(disc_ebn)
    for (old, new) in res
        srp_ebn = update_network!(srp_ebn, old, new)
    end
    ## StructuralReliabilityProblemNode evaluation
    srp_nodes = filter(x -> isa(x, StructuralReliabilityProblemNode), srp_ebn.nodes)
    res2 = map(n -> (n, evaluate!(n)), srp_nodes)
    e_ebn = deepcopy(srp_ebn)
    ## Final Network
    for (old, new) in res2
        e_ebn = update_network!(e_ebn, old, new)
    end
    return e_ebn, [i.name for i in functional_nodes_to_eval]
end


function evaluate!(srp_node::DiscreteStructuralReliabilityProblemNode)
    results = map(zip(keys(srp_node.srps), values(srp_node.srps))) do (comb, srp)
        return (comb, probability_of_failure(srp.models, srp.performance, srp.inputs, srp.simulation))
    end
    f = x -> Dict(Symbol("fail_" * String(srp_node.name)) => x, Symbol("safe_" * String(srp_node.name)) => 1 - x)

    pf = Dict{Vector{Symbol},Dict{Symbol,Real}}()
    cov = Dict{Vector{Symbol},Number}()
    samples = Dict{Vector{Symbol},DataFrame}()
    for (key, res) in results
        pf[key] = f(res[1])
        cov[key] = res[2]
        samples[key] = res[3]
    end
    return DiscreteChildNode(srp_node.name, srp_node.parents, pf, cov, samples, srp_node.parameters)
end

function evaluate!(srp_node::ContinuousStructuralReliabilityProblemNode)
    results = map(zip(keys(srp_node.srps), values(srp_node.srps))) do (comb, srp)
        samples = UncertaintyQuantification.sample(srp.inputs, 1000)
        UncertaintyQuantification.evaluate!(srp.models, samples)
        if isa(srp.models[end], Model)
            res = samples[:, srp.models[end].name]
        elseif isa(srp.models[end], ExternalModel)
            res = samples[:, srp.models[end].output]
        end
        pdf = EmpiricalDistribution(res)
        return (comb, pdf, samples)
    end
    distributions = Dict{Vector{Symbol},UnivariateDistribution}()
    samples = Dict{Vector{Symbol},DataFrame}()

    for (key, pdf, sam) in results
        distributions[key] = pdf
        samples[key] = sam
    end
    return ContinuousChildNode(srp_node.name, srp_node.parents, distributions, samples, srp_node.discretization)
end

function update_network!(
    ebn::EnhancedBayesianNetwork,
    old_node::FunctionalNode,
    new_node::StructuralReliabilityProblemNode
)
    children = get_children(ebn, old_node)
    for child in children
        deleteat!(child.parents, findall(x -> x.name == old_node.name, child.parents)[1])
        push!(child.parents, new_node)
    end
    deleteat!(ebn.nodes, findall(x -> x.name == old_node.name, ebn.nodes)[1])
    push!(ebn.nodes, new_node)
    return EnhancedBayesianNetwork(ebn.nodes)
end

function update_network!(
    ebn::EnhancedBayesianNetwork,
    old_node::StructuralReliabilityProblemNode,
    new_node::ChildNode
)
    children = get_children(ebn, old_node)
    for child in children
        deleteat!(child.parents, findall(x -> x.name == old_node.name, child.parents)[1])
        push!(child.parents, new_node)
    end
    deleteat!(ebn.nodes, findall(x -> x.name == old_node.name, ebn.nodes)[1])
    push!(ebn.nodes, new_node)
    return EnhancedBayesianNetwork(ebn.nodes)
end