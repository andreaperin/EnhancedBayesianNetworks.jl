function evaluate(node::DiscreteFunctionalNode)
    dict = _build_structuralreliabilityproblem(node)
    results = map(zip(keys(dict["srps"]), values(dict["srps"]))) do (comb, srp)
        return (comb, probability_of_failure(srp.models, srp.performance, srp.inputs, srp.simulation))
    end
    f = x -> Dict(Symbol("fail_" * String(dict["name"])) => x, Symbol("safe_" * String(dict["name"])) => 1 - x)
    pf = Dict{Vector{Symbol},Dict{Symbol,Real}}()
    cov = Dict{Vector{Symbol},Number}()
    samples = Dict{Vector{Symbol},DataFrame}()
    for (key, res) in results
        pf[key] = f(res[1])
        cov[key] = res[2]
        samples[key] = res[3]
    end
    return DiscreteChildNode(dict["name"], dict["parents"], pf, cov, samples, dict["parameters"])
end

function evaluate(node::ContinuousFunctionalNode)
    dict = _build_structuralreliabilityproblem(node)
    results = map(zip(keys(dict["srps"]), values(dict["srps"]))) do (comb, srp)
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
    return ContinuousChildNode(dict["name"], dict["parents"], distributions, samples, dict["discretization"])
end

function _build_structuralreliabilityproblem(node::DiscreteFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)

    ancestors = discrete_ancestors(node)
    ancestors_combination = vec(collect(Iterators.product(_get_states.(ancestors)...)))
    ancestors_combination = map(x -> [x...], ancestors_combination)

    srps = map(ancestors_combination) do evidence
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents; init=UQInput[])

        return StructuralReliabilityProblemPMF(node.models, [parameters..., randomvariables...], node.performance, node.simulations)
    end
    srps = Dict(ancestors_combination .=> srps)
    return Dict("name" => node.name, "parents" => ancestors, "srps" => srps, "parameters" => node.parameters)
end

function _build_structuralreliabilityproblem(node::ContinuousFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)

    ancestors = discrete_ancestors(node)
    ancestors_combination = vec(collect(Iterators.product(_get_states.(ancestors)...)))
    ancestors_combination = map(x -> [x...], ancestors_combination)

    srps = map(ancestors_combination) do evidence
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents; init=UQInput[])

        return StructuralReliabilityProblemPDF(node.models, [parameters..., randomvariables...], node.simulations)
    end
    srps = Dict(ancestors_combination .=> srps)

    return Dict("name" => node.name, "parents" => ancestors, "srps" => srps, "discretization" => node.discretization)
end