struct StructuralReliabilityProblemPMF
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    performance::Function
    simulation::AbstractSimulation
end

struct StructuralReliabilityProblemPDF
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    simulation::MonteCarlo
end
struct DiscreteStructuralReliabilityProblemNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},Union{StructuralReliabilityProblemPMF,StructuralReliabilityProblemPDF}}
    parameters::Dict{Symbol,Vector{Parameter}}
end

struct ContinuousStructuralReliabilityProblemNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},Union{StructuralReliabilityProblemPMF,StructuralReliabilityProblemPDF}}
    discretization::AbstractDiscretization
end

const global StructuralReliabilityProblemNode = Union{DiscreteStructuralReliabilityProblemNode,ContinuousStructuralReliabilityProblemNode}