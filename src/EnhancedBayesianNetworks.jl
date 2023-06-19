module EnhancedBayesianNetworks

using Distributions
using GraphRecipes
using Graphs
using LinearAlgebra
using Reexport


@reexport using Graphs
@reexport using UncertaintyQuantification

import Base: *, sum


# Types
export AbstractNode
export ContinuousNode
export DiscreteNode

# struct
export BayesianNetwork
export ConditionalProbabilityDistribution
export ContinuousFunctionalNode
export ContinuousRootNode
export ContinuousStandardNode
export DiscreteFunctionalNode
export DiscreteRootNode
export DiscreteStandardNode
export EnhancedBayesianNetwork
export Factor
export FunctionalNode
export InferenceState
export RootNode
export ReducedBayesianNetwork
export StandardNode

# Constants
const Evidence = Dict{Symbol,Symbol}
export Evidence

# Methods
export evaluate_ebn
export factorize_cpd
export get_children
export get_cpd
export get_models
export get_neighbors
export get_parameters
export get_parents
export get_performance
export get_simulation
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

export pdf
export cdf
export logpdf


include("nodes/nodes.jl")
include("util/wrap.jl")
include("util/interval_verification.jl")
include("util/node_verification.jl")
include("ebn/ebn.jl")
include("util/evidence_verification.jl")
include("inference/inference.jl")

end