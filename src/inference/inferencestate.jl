@auto_hash_equals struct PreciseInferenceState <: AbstractInferenceState
    bn::BayesianNetwork
    query::Vector{Symbol}
    evidence::Evidence

    function PreciseInferenceState(bn::BayesianNetwork, query::Vector{Symbol}, evidence::Evidence)
        _ensure_query_nodes_in_bn_and_not_in_evidence(query, bn.nodes, evidence)
        _verify_evidence(evidence, bn)
        return new(bn, query, evidence)
    end
end

@auto_hash_equals struct ImpreciseInferenceState <: AbstractInferenceState
    cn::CredalNetwork
    query::Vector{Symbol}
    evidence::Evidence
end

function PreciseInferenceState(bn::BayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence)
    return PreciseInferenceState(bn, wrap(query), evidence)
end

function ImpreciseInferenceState(cn::CredalNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence)
    return ImpreciseInferenceState(cn, wrap(query), evidence)
end

function _ensure_query_nodes_in_bn_and_not_in_evidence(query::Vector{Symbol}, nodes::Vector{<:AbstractNode}, evidence::Evidence)
    isempty(query) && return

    q = first(query)
    q ∉ [i.name for i in nodes] && error("Query $q is not in reduced bayesian network")
    q ∈ [i for i in keys(evidence)] && error("Query $q is part of the evidence")

    return _ensure_query_nodes_in_bn_and_not_in_evidence(query[2:end], nodes, evidence)
end