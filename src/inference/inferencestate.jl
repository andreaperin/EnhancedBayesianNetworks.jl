
struct InferenceState
    rbn::ReducedBayesianNetwork
    query::Vector{Symbol}
    evidence::Vector{Tuple{Symbol,N}} where {N<:AbstractNode}

    function InferenceState(rbn::ReducedBayesianNetwork, query::Vector{Symbol}, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
        _ensure_query_nodes_in_pgm_and_not_in_evidence(query, rbn.nodes, evidence)
        return new(rbn, query, evidence)
    end
end

function InferenceState(rbn::ReducedBayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Union{Tuple{Symbol,N},Vector{Tuple{Symbol,N}}}) where {N<:AbstractNode}
    return InferenceState(rbn, wrap(query), wrap(evidence))
end

function _ensure_query_nodes_in_pgm_and_not_in_evidence(query::Vector{Symbol}, nodes::Vector{<:AbstractNode}, evidence::Vector{Tuple{Symbol,N}}) where {N<:AbstractNode}
    isempty(query) && return

    q = first(query)
    q ∉ [i.name for i in nodes] && throw(ArgumentError("Query $q is not in reduced bayesian network"))
    q ∈ [i[2].name for i in evidence] && throw(ArgumentError("Query $q is part of the evidence"))

    return _ensure_query_nodes_in_pgm_and_not_in_evidence(query[2:end], nodes, evidence)
end

# function Base.show(io::IO, inf::InferenceState)
#     println(io, "Query: $(inf.query)")
#     println(io, "Evidence:")
#     for (k, v) in inf.evidence
#         println(io, "  $k => $v")
#     end
# end

"""
infer(InferenceMethod, InferenceState)
Infer p(query|evidence)
"""
infer(rbn::ReducedBayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Union{Tuple{Symbol,N},Vector{Tuple{Symbol,N}}}) where {N<:AbstractNode} = infer(InferenceState(rbn, query, evidence))