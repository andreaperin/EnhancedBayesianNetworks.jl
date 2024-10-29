module EnhancedBayesianNetworks

using AutoHashEquals
using Compose
using DataFrames
using Distributed
using Distributions
# using Graphs
using LinearAlgebra
using NetworkLayout
using Reexport
using SparseArrays
using UncertaintyQuantification: sample, Interval
using Polyhedra: HalfSpace, doubledescription

# @reexport using Graphs
@reexport using UncertaintyQuantification
@reexport using DataFrames
@reexport using SparseArrays


import Base: *, sum, reduce

# Types
export AbstractContinuousInput
export AbstractDiscreteProbability
export AbstractDiscretization
export ContinuousChildNode
export ContinuousFunctionalNode
export ContinuousRootNode
export DiscreteChildNode
export DiscreteFunctionalNode
export DiscreteRootNode
# export AbstractNetwork
# export AbstractNode
export ApproximatedDiscretization
export UnamedProbabilityBox
# export ContinuousNode
# export DiscreteNode
export ExactDiscretization
export EnhancedBayesianNetwork
export AbstractNode
export ContinuousNode
export DiscreteNode
export RootNode
export ChildNode
export FunctionalNode
# struct
# export BayesianNetwork
# export ChildNode
# export ConditionalProbabilityDistribution
# export ContinuousFunctionalNode
# export ContinuousRootNode
# export ContinuousChildNode
# export CredalNetwork
# export DiscreteFunctionalNode
# export DiscreteRootNode
# export DiscreteChildNode
# export EnhancedBayesianNetwork
# export Factor
# export FunctionalNode
# export ImpreciseInferenceState
# export PreciseInferenceState
# export RootNode

# Constants
# const Evidence = Dict{Symbol,Symbol}
# export Evidence

# Methods
export add_child!
export order_net!
export get_parents
# export evaluate
# export evaluate_with_envelope
# export factorize_cpd
# export discrete_ancestors
# export get_children
# export get_cpd
# export get_models
# export get_neighbors
# export _get_parameters
# export get_parents
# export get_performance
# export get_simulation
# export state_combinations
# export get_state_probability
# export _get_continuous_input
# export gplot
# export infer
# export markov_blanket
# export markov_envelope
# export saveplot
# export show

include("nodes/nodes.jl")
include("ebn/ebn.jl")
include("util/base_show.jl")

end