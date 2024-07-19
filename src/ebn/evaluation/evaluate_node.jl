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
        additional_info = Dict{Vector{Symbol},Dict}()
        for evidence in ancestors_combination
            parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
            randomvariables = mapreduce(p -> get_continuous_input(p, evidence), vcat, continuous_parents; init=UQInput[])
            df = UncertaintyQuantification.sample([parameters..., randomvariables...], node.simulation)
            UncertaintyQuantification.evaluate!(node.models, df)
            pdf = EmpiricalDistribution(df[:, node.models[end].name])
            distribution[evidence] = pdf
            additional_info[evidence] = Dict(:samples => df)
        end
        if isempty(ancestors)
            return ContinuousRootNode(node.name, distribution[[]], additional_info[[]], ExactDiscretization(node.discretization.intervals))
        else
            return ContinuousChildNode(node.name, ancestors, distribution, additional_info, node.discretization)
        end
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
    additional_info = Dict{Vector{Symbol},Dict}()
    # cov = Dict()
    # samples = Dict{Vector{Symbol},DataFrame}()
    for evidence in ancestors_combination
        parameters = mapreduce(p -> get_parameters(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> get_continuous_input(p, evidence), vcat, continuous_parents; init=UQInput[])
        res = probability_of_failure(node.models, node.performance, [parameters..., randomvariables...], node.simulation)
        if isa(node.simulation, Union{AbstractMonteCarlo,LineSampling,ImportanceSampling,UncertaintyQuantification.AbstractSubSetSimulation})
            pf[evidence] = f(res[1])
            additional_info[evidence] = Dict(:cov => res[2], :samples => res[3])
        elseif isa(node.simulation, DoubleLoop)
            pf[evidence] = f_interval(res)
            additional_info[evidence] = Dict()
        elseif isa(node.simulation, RandomSlicing)
            pf[evidence] = f_interval(res[1])
            additional_info[evidence] = Dict(:lb => res[2], :ub => res[3])
        elseif isa(node.simulation, FORM)
            pf[evidence] = f(res[1])
            additional_info[evidence] = Dict(:β => res[2], :design_point => res[3], :α => res[4])
        end
    end
    if isempty(ancestors)
        return DiscreteRootNode(node.name, pf[[]], additional_info[[]], node.parameters)
    else
        return DiscreteChildNode(node.name, ancestors, pf, additional_info, node.parameters)
    end
end