module EnhancedBayesianNetworks

using AutoHashEquals
using DataFrames
using Distributed
using Distributions
using GraphRecipes
using Graphs
using LinearAlgebra
using Reexport
using UncertaintyQuantification: sample

@reexport using Graphs
@reexport using UncertaintyQuantification
@reexport using DataFrames

import Base: *, sum, reduce

# Types
export AbstractNode
export AbstractDiscretization
export ApproximatedDiscretization
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
export DiscreteFunctionalNode
export DiscreteRootNode
export DiscreteChildNode
export EnhancedBayesianNetwork
export Factor
export FunctionalNode
export InferenceState
export RootNode

# Constants
const Evidence = Dict{Symbol,Symbol}
export Evidence

# Methods
export discretize
export evaluate
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
export get_randomvariable
export infer
export is_equal
export markov_blanket
export markov_envelope
export minimal_increase_in_complexity
export reducedim
export reducedim!
export reduce_ebn_markov_envelopes
export reduce_ebn_standard
export show
export transfer_continuous
export update_network!

export pdf
export cdf
export logpdf

include("util/wrap.jl")
include("nodes/nodes.jl")
include("ebn/ebn.jl")
include("util/plots.jl")
# include("util/evidence_verification.jl")
# include("inference/inference.jl")

end