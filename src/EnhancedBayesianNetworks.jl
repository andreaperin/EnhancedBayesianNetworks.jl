module EnhancedBayesianNetworks

using AutoHashEquals
using Compose
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
export AbstractNode
export ApproximatedDiscretization
export UnamedProbabilityBox
export ContinuousNode
export DiscreteNode
export ExactDiscretization
export EnhancedBayesianNetwork
export AbstractNode
export ContinuousNode
export DiscreteNode
export RootNode
export ChildNode
export FunctionalNode
export add_child!
export order_net!
export get_parents
export get_children
export markov_blanket
export markov_envelope
export gplot
export saveplot
export evaluate!
export evaluate_with_envelopes
export reduce!

include("util/wrap.jl")
include("nodes/nodes.jl")
include("ebn/ebn.jl")
include("util/base_show.jl")
include("util/plots.jl")

end