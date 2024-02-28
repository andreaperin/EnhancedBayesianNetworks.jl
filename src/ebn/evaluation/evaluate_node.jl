function _evaluate(node::ContinuousFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    ancestors = discrete_ancestors(node)
    ancestors_combination = vec(collect(Iterators.product(_get_states.(ancestors)...)))
    ancestors_combination = map(x -> [x...], ancestors_combination)
    distributions = Dict{Vector{Symbol},UnivariateDistribution}()
    samples = Dict{Vector{Symbol},DataFrame}()
    for evidence in ancestors_combination
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents; init=UQInput[])
        df = UncertaintyQuantification.sample([parameters..., randomvariables...], node.simulation)
        UncertaintyQuantification.evaluate!(node.models, df)
        pdf = EmpiricalDistribution(df[:, node.models[end].name])
        distributions[evidence] = pdf
        samples[evidence] = df
    end
    return ContinuousChildNode(node.name, ancestors, distributions, samples, node.discretization)
end

function _evaluate(node::DiscreteFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    ancestors = discrete_ancestors(node)
    ancestors_combination = vec(collect(Iterators.product(_get_states.(ancestors)...)))
    ancestors_combination = map(x -> [x...], ancestors_combination)
    f = x -> Dict(Symbol("fail_$(node.name)") => x, Symbol("safe_$(node.name)") => 1 - x)
    pf = Dict{Vector{Symbol},Dict{Symbol,Real}}()
    cov = Dict{Vector{Symbol},Number}()
    samples = Dict{Vector{Symbol},DataFrame}()
    for evidence in ancestors_combination
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_randomvariable(p, evidence), vcat, continuous_parents; init=UQInput[])
        res = probability_of_failure(node.models, node.performance, [parameters..., randomvariables...], node.simulation)
        pf[evidence] = f(res[1])
        cov[evidence] = res[2]
        samples[evidence] = res[3]
    end
    return DiscreteChildNode(node.name, ancestors, pf, cov, samples, node.parameters)
end