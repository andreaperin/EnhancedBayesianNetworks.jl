function infer(inf::InferenceState)
    bn = inf.bn
    nodes = bn.nodes
    query = inf.query
    evidence = inf.evidence
    hidden = setdiff([i.name for i in nodes], vcat(query, collect(keys(evidence))))
    ##TODO add algo for optimize the elimination order
    factors = map(n -> Factor(bn, n.name, evidence), nodes)
    # successively remove the hidden nodes
    for h in hidden
        contain_h = filter(ϕ -> h in ϕ, factors)
        if !isempty(contain_h)
            factors = setdiff(factors, contain_h)
            τ_h = sum(reduce((*), contain_h), h)
            push!(factors, τ_h)
        end
    end
    ϕ = reduce((*), factors)
    tot = sum(abs, ϕ.potential)
    ϕ.potential ./= tot
    return ϕ
end

infer(bn::BayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence=Evidence()) = infer(InferenceState(bn, query, evidence))