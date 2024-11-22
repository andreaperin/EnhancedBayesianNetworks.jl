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
export ContinuousFunctionalNode
export DiscreteFunctionalNode
export ApproximatedDiscretization
export UnamedProbabilityBox
export ContinuousNode
export DiscreteNode
export FunctionalNode
export ExactDiscretization
export EnhancedBayesianNetwork
export AbstractNode
export ContinuousNode
export BayesianNetwork
export CredalNetwork
export Factor
export AbstractInferenceState
export PreciseInferenceState
export ImpreciseInferenceState
export ConditionalProbabilityDistribution


const Evidence = Dict{Symbol,Symbol}
export Evidence


export add_child!
# export order!
export parents
export children
export discrete_ancestors
export gplot
# export markov_blanket
# export markov_envelope
# export gplot
# export saveplot
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