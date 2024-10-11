module EnhancedBayesianNetworks

using AutoHashEquals
using DataFrames
using Distributed
using Distributions
using GraphRecipes
using Graphs
using LinearAlgebra
using Reexport
using UncertaintyQuantification: sample, Interval
using Polyhedra: HalfSpace, doubledescription

@reexport using Graphs
@reexport using UncertaintyQuantification
@reexport using DataFrames

import Base: *, sum, reduce

# Types
export AbstractNode
export AbstractDiscretization
export ApproximatedDiscretization
export UnamedProbabilityBox
export ContinuousNode
export DiscreteNode
export ExactDiscretization

# struct
export BayesianNetwork
export ChildNode
export ConditionalProbabilityDistribution
export ContinuousFunctionalNode
export ContinuousRootNode
export ContinuousChildNode
export CredalNetwork
export DiscreteFunctionalNode
export DiscreteRootNode
export DiscreteChildNode
export EnhancedBayesianNetwork
export Factor
export FunctionalNode
export ImpreciseInferenceState
export PreciseInferenceState
export RootNode

# Constants
const Evidence = Dict{Symbol,Symbol}
export Evidence

# Methods
export evaluate
export evaluate_with_envelope
export factorize_cpd
export discrete_ancestors
export get_children
export get_cpd
export get_models
export get_neighbors
export get_parameters
export get_parents
export get_performance
export get_simulation
export state_combinations
export get_state_probability
export get_continuous_input
export infer
export markov_blanket
export markov_envelope
export show

include("nodes/nodes.jl")
include("ebn/ebn.jl")
include("inference/inference.jl")

end