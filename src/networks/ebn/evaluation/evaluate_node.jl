function _evaluate_node(net::EnhancedBayesianNetwork, node::ContinuousFunctionalNode, collect_samples::Bool=true)

    function _get_evidence_from_state(s::Symbol)
        return filter(n -> s ∈ states(n), filter(x -> isa(x, DiscreteNode), net.nodes))[1].name, s
    end

    if all(isprecise.(parents(net, node)[3]))
        ancestors = discrete_ancestors(net, node)
        ancestors_combination = sort(vec(collect(Iterators.product(states.(ancestors)...))))
        evidences = map(ac -> Evidence(_get_evidence_from_state.(ac)), ancestors_combination)

        if isempty(ancestors_combination[1])
            cpt = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
            discretization = ExactDiscretization(node.discretization.intervals)
        else
            cpt = ContinuousConditionalProbabilityTable{PreciseContinuousInput}([i.name for i in ancestors])
            discretization = node.discretization
        end

        add_info = Dict{Vector{Symbol},Dict}()

        for evidence in evidences
            inputs_uq = mapreduce(p -> _uq_inputs(p, evidence), vcat, parents(net, node)[3]; init=UQInput[])
            df = UncertaintyQuantification.sample(inputs_uq, node.simulation)
            UncertaintyQuantification.evaluate!(node.models, df)
            pdf = EmpiricalDistribution(df[:, node.models[end].name])

            cpt[evidence] = pdf

            if collect_samples
                add_info[collect(values(evidence))] = Dict(:samples => df)
            end
        end

        return ContinuousNode(node.name, cpt, discretization, add_info)
    else
        error("node $(node.name) is a continuousfunctionalnode with at least one parent with Interval or p-boxes in its distributions. No method for extracting failure probability p-box have been implemented yet")
    end
end

function _evaluate_node(net::EnhancedBayesianNetwork, node::DiscreteFunctionalNode, collect_samples::Bool=true)

    function _get_evidence_from_state(s::Symbol)
        node_names = filter(n -> s ∈ states(n), ancestors)
        if !isempty(node_names)
            return node_names[1].name, s
        else
            return node.name, s
        end
    end

    ancestors = discrete_ancestors(net, node)

    ancestors = discrete_ancestors(net, node)
    ancestors_combination = sort(vec(collect(Iterators.product(states.(ancestors)...))))
    evidences = map(ac -> Evidence(_get_evidence_from_state.(ac)), ancestors_combination)

    if all(isprecise.(parents(net, node)[3]))
        if isempty(ancestors_combination[1])
            cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(node.name)
            evidences = [Evidence()]
        else
            cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(vcat([i.name for i in ancestors], node.name))
        end
    else
        if isempty(ancestors_combination[1])
            cpt = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(node.name)
            evidences = [Evidence()]
        else
            cpt = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(vcat([i.name for i in ancestors], node.name))
        end
    end

    additional_info = Dict{AbstractVector{Symbol},Dict}()

    for evidence in evidences

        input_uq = mapreduce(p -> _uq_inputs(p, evidence), vcat, parents(net, node)[3]; init=UQInput[])

        res = probability_of_failure(node.models, node.performance, input_uq, node.simulation)

        fail_ev = deepcopy(evidence)
        fail_ev[node.name] = Symbol(string(node.name) * "_fail")
        safe_ev = deepcopy(evidence)
        safe_ev[node.name] = Symbol(string(node.name) * "_safe")

        if isa(node.simulation, Union{AbstractMonteCarlo,LineSampling,ImportanceSampling,UncertaintyQuantification.AbstractSubSetSimulation})
            cpt[fail_ev] = res[1]
            cpt[safe_ev] = 1 - res[1]
            if collect_samples
                additional_info[collect(values(evidence))] = Dict(:cov => res[2], :samples => res[3])
            end

        elseif isa(node.simulation, FORM)
            cpt[fail_ev] = res[1]
            cpt[safe_ev] = 1 - res[1]
            if collect_samples
                additional_info[collect(values(evidence))] = Dict(:β => res[2], :design_point => res[3], :α => res[4])
            end

        elseif isa(node.simulation, DoubleLoop)
            !isa(res, Interval) ? res = Interval(res, res, :pf) : nothing
            cpt[fail_ev] = (res.lb, res.ub)
            cpt[safe_ev] = (1 - res.ub, 1 - res.lb)
            if collect_samples
                additional_info[collect(values(evidence))] = Dict()
            end
        elseif isa(node.simulation, RandomSlicing)
            !isa(res[1], Interval) ? res[1] = Interval(res[1], res[1], :pf) : nothing
            cpt[fail_ev] = (res[1].lb, res[1].ub)
            cpt[safe_ev] = (1 - res[1].ub, 1 - res[1].lb)
            if collect_samples
                additional_info[collect(values(evidence))] = Dict(:lb => res[2], :ub => res[3])
            end
        end
    end

    return DiscreteNode(node.name, cpt, node.parameters, additional_info)
end