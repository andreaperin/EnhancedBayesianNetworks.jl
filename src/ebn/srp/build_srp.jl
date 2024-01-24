function _build_structuralreliabilityproblem(node::DiscreteFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)

    ancestors = discrete_ancestors(node)
    ancestors_combination = vec(collect(Iterators.product(EnhancedBayesianNetworks._get_states.(ancestors)...)))
    ancestors_combination = map(x -> [x...], ancestors_combination)

    srps = map(ancestors_combination) do evidence
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents; init=UQInput[])

        return EnhancedBayesianNetworks.StructuralReliabilityProblemPMF(node.models, [parameters..., randomvariables...], node.performance, node.simulations)
    end
    srps = Dict(ancestors_combination .=> srps)
    return EnhancedBayesianNetworks.DiscreteStructuralReliabilityProblemNode(node.name, ancestors, srps, node.parameters)
end

function _build_structuralreliabilityproblem(node::ContinuousFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)

    ancestors = discrete_ancestors(node)
    ancestors_combination = vec(collect(Iterators.product(EnhancedBayesianNetworks._get_states.(ancestors)...)))
    ancestors_combination = map(x -> [x...], ancestors_combination)

    srps = map(ancestors_combination) do evidence
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents; init=UQInput[])

        return EnhancedBayesianNetworks.StructuralReliabilityProblemPDF(node.models, [parameters..., randomvariables...], node.simulations)
    end
    srps = Dict(ancestors_combination .=> srps)

    return EnhancedBayesianNetworks.ContinuousStructuralReliabilityProblemNode(node.name, ancestors, srps, node.discretization)
end