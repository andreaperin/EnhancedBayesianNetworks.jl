module EnhancedBayesianNetworks

using AutoHashEquals
using DataFrames
using Distributed
using Distributions
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
@reexport using Compose

import Base: *, sum, reduce

# Types
export AbstractContinuousInput
export AbstractDiscreteProbability
export AbstractDiscretization
export AbstractInferenceState
export AbstractNetwork
export AbstractNode
export AbstractContinuousNode
export AbstractDiscreteNode
export ApproximatedDiscretization
export BayesianNetwork
export ConditionalProbabilityDistribution
export ContinuousFunctionalNode
export ContinuousNode
# export CredalNetwork
export DiscreteFunctionalNode
export DiscreteNode
export ExactDiscretization
export EnhancedBayesianNetwork
export Evidence
export FunctionalNode
export ImpreciseInferenceState
export PreciseInferenceState
export UnamedProbabilityBox
# export Factor

## Constants
const Evidence = Dict{Symbol,Symbol}

## Functions
export add_child!
export children
export discrete_ancestors
export evaluate!
export evaluate_with_envelopes
export gplot
export markov_blanket
export markov_envelope
export order!
export parents
export reduce!
export saveplot
# export evaluate!
# export evaluate_with_envelopes
# export reduce!
# export get_cpd
# export infer
# export factorize_cpd
# export dispatch_network

include("util/wrap.jl")
include("util/verification_common.jl")
include("nodes/nodes.jl")
include("ebn/ebn.jl")
# include("inference/inference.jl")
include("util/base_show.jl")
include("util/plots.jl")
end