abstract type ProbabilisticGraphicalModel end

include("enhancedbn.jl")
include("reducedbn.jl")
include("bayesnet.jl")
include("discretization/discretize_root.jl")
include("discretization/discretize_child.jl")