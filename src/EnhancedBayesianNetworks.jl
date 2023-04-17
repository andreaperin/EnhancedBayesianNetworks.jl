module EnhancedBayesianNetworks

using Graphs
using Reexport

@reexport using UncertaintyQuantification

export EnhancedBayesNet
export FunctionalCPD
export FunctionalNode
export ModelParameters
export ModelWithName
export NamedCategorical
export RootCPD
export RootNode
export StdCPD
export StdNode

export _build_node_evidence_after_reduction
export _functional_node_after_reduction
export markov_envelopes
export name
export _reduce_ebn_to_rbn

include("CPDs.jl")

include("nodes/nodes.jl")
include("bn.jl")

end
