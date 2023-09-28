function _build_structuralreliabilityproblem_node(ebn::EnhancedBayesianNetwork, node::DiscreteFunctionalNode)
    ebn_discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    ebn_continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)

    rbn_discrete_parents = get_discrete_ancestors(node)
    rbn_discrete_parents_combination = vec(collect(Iterators.product(_get_states.(rbn_discrete_parents)...)))
    rbn_discrete_parents_combination = map(x -> [i for i in x], rbn_discrete_parents_combination)
    srps = Dict{Vector{Symbol},StructuralReliabilityProblemPMF}()

    for evidence in rbn_discrete_parents_combination
        if isempty(ebn_discrete_parents)
            uq_parameters = Vector{UQInput}()
        else
            uq_parameters = mapreduce(p -> get_parameters(p, evidence), vcat, ebn_discrete_parents)
        end
        uq_randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, ebn_continuous_parents)
        uqinputs = vcat(uq_parameters, uq_randomvariables)
        simulations = node.simulations
        models = node.models
        performance = node.performance

        srps[evidence] = StructuralReliabilityProblemPMF(models, uqinputs, performance, simulations)
    end
    return DiscreteStructuralReliabilityProblemNode(node.name, rbn_discrete_parents, srps, node.parameters)
end

function _build_structuralreliabilityproblem_node(ebn::EnhancedBayesianNetwork, node::ContinuousFunctionalNode)
    ebn_discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    ebn_continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)

    rbn_discrete_parents = get_discrete_ancestors(node)
    rbn_discrete_parents_combination = vec(collect(Iterators.product(_get_states.(rbn_discrete_parents)...)))
    rbn_discrete_parents_combination = map(x -> [i for i in x], rbn_discrete_parents_combination)
    srps = Dict{Vector{Symbol},StructuralReliabilityProblemPDF}()

    for evidence in rbn_discrete_parents_combination
        if isempty(ebn_discrete_parents)
            uq_parameters = Vector{UQInput}()
        else
            uq_parameters = mapreduce(p -> get_parameters(p, evidence), vcat, ebn_discrete_parents)
        end
        uq_randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, ebn_continuous_parents)
        uqinputs = vcat(uq_parameters, uq_randomvariables)
        simulations = node.simulations
        models = node.models

        srps[evidence] = StructuralReliabilityProblemPDF(models, uqinputs, simulations)
    end
    return ContinuousStructuralReliabilityProblemNode(node.name, rbn_discrete_parents, srps, node.discretization)
end