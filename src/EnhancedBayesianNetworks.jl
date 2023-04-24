module EnhancedBayesianNetworks

using GraphRecipes
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
export DiscreteFunctionalNode
export DiscreteRootNode
export DiscreteStandardNode
export EnhancedBayesianNetwork
export RootNode
export StandardNode


# Methods
export show

include("nodes/nodes.jl")
include("util/probabilities.jl")
include("ebn.jl")

end
