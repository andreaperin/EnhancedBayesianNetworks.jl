
struct InferenceState
    bn::BayesianNetwork
    query::Vector{Symbol}
    evidence::Evidence

    function InferenceState(bn::BayesianNetwork, query::Vector{Symbol}, evidence::Evidence)
        _ensure_query_nodes_in_bn_and_not_in_evidence(query, bn.nodes, evidence)
        verify_evidence(evidence, bn)
        return new(bn, query, evidence)
    end
end

function InferenceState(bn::BayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence)
    return InferenceState(bn, wrap(query), evidence)
end

function _ensure_query_nodes_in_bn_and_not_in_evidence(query::Vector{Symbol}, nodes::Vector{<:AbstractNode}, evidence::Evidence)
    isempty(query) && return

    q = first(query)
    q ∉ [i.name for i in nodes] && error("Query $q is not in reduced bayesian network")
    q ∈ [i for i in keys(evidence)] && error("Query $q is part of the evidence")

    return _ensure_query_nodes_in_bn_and_not_in_evidence(query[2:end], nodes, evidence)
end

function Base.show(io::IO, inf::InferenceState)
    println(io, "Query: $(inf.query)")
    println(io, "Evidence:")
    for (k, v) in inf.evidence
        println(io, "  $k => $v")
    end
end

