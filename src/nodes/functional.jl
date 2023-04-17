mutable struct FunctionalNode <: AbstractNode
    cpd::FunctionalCPD
    parents::Vector{<:AbstractNode}
    type::String
    evidence_table::Vector{EvidenceTable}
end

function FunctionalNode(cpd::FunctionalCPD, parents::Vector{<:AbstractNode}, type::String)
    evidence_table = _build_evidencetable_from_cpd(cpd, parents)
    FunctionalNode(cpd, parents, type, evidence_table)
end