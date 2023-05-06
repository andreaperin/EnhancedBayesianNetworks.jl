module EnhancedBayesianNetworks

using GraphRecipes
using Graphs
using Reexport


@reexport using Graphs
@reexport using OrderedCollections
@reexport using UncertaintyQuantification


# Types
export AbstractNode
export ContinuousNode
export DiscreteNode

# struct
export ContinuousRootNode
export ContinuousStandardNode
export DiscreteFunctionalNode
export DiscreteRootNode
export DiscreteStandardNode
export EnhancedBayesianNetwork
export RootNode
export ReducedBayesianNetwork
export StandardNode
export StructuralReliabilityProblem
export StructuralReliabilityProblemNode


# Methods
export get_children
export get_models
export get_neighbors
export get_parameters
export get_parents
export get_state_probability
export get_randomvariable
export markov_blanket
export markov_envelope
export reduce_ebn_markov_envelopes
export reduce_ebn_standard
export show


include("nodes/nodes.jl")
include("util/node_verification.jl")
include("ebn/ebn.jl")

end