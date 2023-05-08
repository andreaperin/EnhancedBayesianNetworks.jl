struct StructuralReliabilityProblem
    models::Vector{<:UQModel}
    inputs::Vector{<:UQInput}
    performance::Function
    simulation::AbstractSimulation
end

struct StructuralReliabilityProblemNode
    name::Symbol
    parents::Vector{<:AbstractNode}
    srps::OrderedDict{Vector{Symbol},StructuralReliabilityProblem}
end
