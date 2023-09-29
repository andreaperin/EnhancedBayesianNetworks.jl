struct StructuralReliabilityProblemPDF
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    simulation::MonteCarlo
end
struct StructuralReliabilityProblemPMF
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    performance::Function
    simulation::AbstractSimulation
end

struct ContinuousStructuralReliabilityProblemNode <: ContinuousNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},StructuralReliabilityProblemPDF}
    discretization::AbstractDiscretization
end

struct DiscreteStructuralReliabilityProblemNode <: DiscreteNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::Dict{Vector{Symbol},StructuralReliabilityProblemPMF}
    parameters::Dict{Symbol,Vector{Parameter}}
end

const global StructuralReliabilityProblemNode = Union{DiscreteStructuralReliabilityProblemNode,ContinuousStructuralReliabilityProblemNode}