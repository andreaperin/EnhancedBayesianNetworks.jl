module EnhancedBayesianNetworks

using Graphs
using Reexport

@reexport using UncertaintyQuantification
@reexport using OrderedCollections

# Types
export AbstractNode
export ContinuousNode
export DiscreteNode

# struct
export ContinuousRootNode
export ContinuousStandardNode
# export DiscreteFunctionalNode
export DiscreteRootNode
export DiscreteStandardNode
export RootNode
export StandardNode


# Methods
# export _get_states


include("nodes/nodes.jl")
include("util/probabilities.jl")
end
