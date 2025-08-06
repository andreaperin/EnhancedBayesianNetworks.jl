abstract type AbstractNetwork end

include("../util/verification_add_child.jl")
include("networks_common.jl")
include("ebn/ebn.jl")
include("bn/bayesnet.jl")
include("bn/bayesnet2be.jl")
include("cn/credalnet.jl")
include("dispatch.jl")