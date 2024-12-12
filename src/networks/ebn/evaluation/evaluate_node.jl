function _evaluate_node(net::EnhancedBayesianNetwork, node::ContinuousFunctionalNode)
    if all(_is_precise.(parents(net, node)[3]))
        discrete_parents = filter(x -> isa(x, DiscreteNode), parents(net, node)[3])
        continuous_parents = filter(x -> isa(x, ContinuousNode), parents(net, node)[3])
        ancestors = discrete_ancestors(net, node)

        ancestors_combination = sort(vec(collect(Iterators.product(_states.(ancestors)...))))
        if isempty(ancestors_combination[1])
            new_cpt = DataFrame()
            evidences = [Evidence()]
            discretization = ExactDiscretization(node.discretization.intervals)
        else
            new_cpt = DataFrame(ancestors_combination, [i.name for i in ancestors])
            evidences = map(x -> Dict(pairs(new_cpt[x, :])), range(1, nrow(new_cpt)))
            discretization = node.discretization
        end

        dists = []
        add_info = Dict{Vector{Symbol},Dict}()
        for evidence in evidences
            parameters = mapreduce(p -> _parameters_with_evidence(p, evidence), vcat, discrete_parents; init=UQInput[])
            randomvariables = mapreduce(p -> _continuous_input(p, evidence), vcat, continuous_parents; init=UQInput[])
            df = UncertaintyQuantification.sample([parameters..., randomvariables...], node.simulation)
            UncertaintyQuantification.evaluate!(node.models, df)
            pdf = EmpiricalDistribution(df[:, node.models[end].name])
            push!(dists, pdf)
            add_info[collect(values(evidence))] = Dict(:samples => df)
        end

        new_cpt[!, :Prob] = dists

        return ContinuousNode{UnivariateDistribution}(node.name, new_cpt, discretization, add_info)
    else
        error("node $(node.name) is a continuousfunctionalnode with at least one parent with Interval or p-boxes in its distributions. No method for extracting failure probability p-box have been implemented yet")
    end
end

function _evaluate_node(net::EnhancedBayesianNetwork, node::DiscreteFunctionalNode)
    discrete_parents = filter(x -> isa(x, DiscreteNode), parents(net, node)[3])
    continuous_parents = filter(x -> isa(x, ContinuousNode), parents(net, node)[3])
    ancestors = discrete_ancestors(net, node)

    ancestors_combination = sort(vec(collect(Iterators.product(_states.(ancestors)...))))
    node_states = [Symbol(string(node.name) * "_fail"), Symbol(string(node.name) * "_safe")]

    if isempty(ancestors_combination[1])
        new_cpt = DataFrame()
        evidences = [Evidence()]
        new_cpt[!, node.name] = node_states
    else
        new_cpt = DataFrame(ancestors_combination, [i.name for i in ancestors])
        evidences = map(x -> Dict(pairs(new_cpt[x, :])), range(1, nrow(new_cpt)))
        node_states = repeat([Symbol(string(node.name) * "_fail"), Symbol(string(node.name) * "_safe")], nrow(new_cpt))
        new_cpt = repeat(new_cpt, inner=2)
        new_cpt[!, node.name] = node_states
    end

    additional_info = Dict{AbstractVector{Symbol},Dict}()
    probs = []
    for evidence in evidences
        parameters = mapreduce(p -> _parameters_with_evidence(p, evidence), vcat, discrete_parents; init=UQInput[])
        randomvariables = mapreduce(p -> _continuous_input(p, evidence), vcat, continuous_parents; init=UQInput[])
        res = probability_of_failure(node.models, node.performance, [parameters..., randomvariables...], node.simulation)

        if isa(node.simulation, Union{AbstractMonteCarlo,LineSampling,ImportanceSampling,UncertaintyQuantification.AbstractSubSetSimulation})
            push!(probs, res[1])
            push!(probs, 1 - res[1])
            additional_info[collect(values(evidence))] = Dict(:cov => res[2], :samples => res[3])
        elseif isa(node.simulation, DoubleLoop)
            !isa(res, Interval) ? res = Interval(res, res, :pf) : nothing
            push!(probs, [res.lb, res.ub])
            push!(probs, [1 - res.ub, 1 - res.lb])
            additional_info[collect(values(evidence))] = Dict()
        elseif isa(node.simulation, RandomSlicing)
            !isa(res[1], Interval) ? res[1] = Interval(res[1], res[1], :pf) : nothing
            push!(probs, [res[1].lb, res[1].ub])
            push!(probs, [1 - res[1].ub, 1 - res[1].lb])
            additional_info[collect(values(evidence))] = Dict(:lb => res[2], :ub => res[3])
        elseif isa(node.simulation, FORM)
            push!(probs, res[1])
            push!(probs, 1 - res[1])
            additional_info[collect(values(evidence))] = Dict(:β => res[2], :design_point => res[3], :α => res[4])
        end
    end

    new_cpt[!, :Prob] = probs

    return DiscreteNode(node.name, new_cpt, node.parameters, additional_info)
end