function _evaluate(node::ContinuousFunctionalNode)
    if any(_is_imprecise.(node.parents))
        error("node $(node.name) is a continuousfunctionalnode with at least one parent with Interval or p-boxes in its distributions. No method for extracting failure probability p-box have been implemented yet")
    else
        discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
        continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
        ancestors = discrete_ancestors(node)
        ancestors_combination = vec(collect(Iterators.product(_get_states.(ancestors)...)))
        ancestors_combination = map(x -> [x...], ancestors_combination)
        distribution = Dict{Vector{Symbol},UnivariateDistribution}()
        samples = Dict{Vector{Symbol},DataFrame}()
        for evidence in ancestors_combination
            parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
            randomvariables = mapreduce(p -> get_continuous_input(p, evidence), vcat, continuous_parents; init=UQInput[])
            df = UncertaintyQuantification.sample([parameters..., randomvariables...], node.simulation)
            UncertaintyQuantification.evaluate!(node.models, df)
            pdf = EmpiricalDistribution(df[:, node.models[end].name])
            distribution[evidence] = pdf
            samples[evidence] = df
        end
        return ContinuousChildNode(node.name, ancestors, distribution, samples, node.discretization)
    end
end

function _evaluate(node::DiscreteFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), node.parents)
    continuous_parents = filter(x -> isa(x, ContinuousNode), node.parents)
    ancestors = discrete_ancestors(node)
    ancestors_combination = vec(collect(Iterators.product(_get_states.(ancestors)...)))
    ancestors_combination = map(x -> [x...], ancestors_combination)
    f = x -> convert(Dict{Symbol,Real}, Dict(Symbol("fail_$(node.name)") => x, Symbol("safe_$(node.name)") => 1 - x))
    f_interval = x -> convert(Dict{Symbol,Vector{Real}}, Dict(Symbol("fail_$(node.name)") => [x.lb, x.ub], Symbol("safe_$(node.name)") => [1 - x.ub, 1 - x.lb]))
    pf = Dict()
    cov = Dict()
    samples = Dict{Vector{Symbol},DataFrame}()
    for evidence in ancestors_combination
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_continuous_input(p, evidence), vcat, continuous_parents; init=UQInput[])
        res = probability_of_failure(node.models, node.performance, [parameters..., randomvariables...], node.simulation)
        if isa(res, Tuple{Real,Real,DataFrame})
            pf[evidence] = f(res[1])
            cov[evidence] = res[2]
            samples[evidence] = res[3]
        elseif isa(res, Real)
            pf[evidence] = f(res)
            cov[evidence] = 0
            samples[evidence] = DataFrame()
        elseif isa(res, Interval)
            pf[evidence] = f_interval(res)
            cov[evidence] = 0
            samples[evidence] = DataFrame()
        end
    end
    return DiscreteChildNode(node.name, ancestors, pf, cov, samples, node.parameters)
end