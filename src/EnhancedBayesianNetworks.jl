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
export AbstractContinuousNode
export AbstractDiscreteNode
export AbstractDiscretization
export AbstractInferenceState
export AbstractNetwork
export AbstractNode
export ApproximatedDiscretization
export BayesianNetwork
export BayesianNetwork2be
export ContinuousFunctionalNode
export ContinuousInput
export ContinuousNode
export ContinuousConditionalProbabilityTable
export CredalNetwork
export DiscreteConditionalProbabilityTable
export DiscreteFunctionalNode
export DiscreteNode
export DiscreteProbability
export ExactDiscretization
export EnhancedBayesianNetwork
export Evidence
export Factor
export FunctionalNode
export ImpreciseInferenceState
export ImpreciseContinuousInput
export ImpreciseDiscreteProbability
export PreciseContinuousInput
export PreciseDiscreteProbability
export PreciseInferenceState
export UnamedProbabilityBox
# export Factor

## Constants
const Evidence = Dict{Symbol,Symbol}

## Functions
export add_child!
export children
export discrete_ancestors
export dispatch
export distributions
export evaluate!
export evaluate_with_envelopes
export factorize
export gplot
export infer
export isprecise
export isroot
export joint_probability
export learn_parameters_EM
export learn_parameters_MLE
export markov_blanket
export markov_envelope
export order!
export parents
export reduce!
export saveplot
export scenarios
export states

include("util/wrap.jl")
include("nodes/nodes.jl")
include("networks/networks.jl")
include("inference/inference.jl")
include("learning/learning.jl")

include("util/base_show.jl")
include("util/plots.jl")
end