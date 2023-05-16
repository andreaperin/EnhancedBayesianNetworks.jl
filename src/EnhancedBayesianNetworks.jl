module EnhancedBayesianNetworks

using Distributions
using GraphRecipes
using Graphs
using LinearAlgebra
using Reexport


@reexport using Graphs
@reexport using OrderedCollections
@reexport using UncertaintyQuantification


# Types
export AbstractNode
export ContinuousNode
export DiscreteNode

# struct
export ContinuousFunctionalNode
export ContinuousRootNode
export ContinuousStandardNode
export DiscreteFunctionalNode
export DiscreteRootNode
export DiscreteStandardNode
export EnhancedBayesianNetwork
# export EvidenceDistribution
export FunctionalNode
export RootNode
export ReducedBayesianNetwork
export StandardNode
# export StructuralReliabilityProblem
# export StructuralReliabilityProblemNode



# Methods
export evaluate_rbn
export get_children
export get_models
export get_neighbors
export get_parameters
export get_parents
export get_performance
export get_simulation
export get_state_probability
export get_randomvariable
export markov_blanket
export markov_envelope
export reduce_ebn_markov_envelopes
export reduce_ebn_standard
export show

export pdf
export cdf
export logpdf


include("nodes/nodes.jl")
include("util/interval_verification.jl")
include("util/node_verification.jl")
include("ebn/ebn.jl")

end