include("../bn.jl")
Sys.isapple() ? include("../model_TH_macos/buildmodel_TH.jl") : include("../model_TH_win/buildmodel_TH.jl")
include("../CPDs.jl")
include("../nodes.jl")
include("../models_probabilities.jl")
include("../bn.jl")

a = NamedCategorical([:first, :second, :third], [0.34, 0.33, 0.33])
CPDa = RootCPD(:t, a)
t = StdNode(CPDa)

b = NamedCategorical([:happen, :nothappen], [0.5, 0.5])
CPDp = RootCPD(:p, b)
p = StdNode(CPDp)

c = Rayleigh(0.387)
CPDc = RootCPD(:w, c)
w = StdNode(CPDc)

wraising1 = Rayleigh(0.387)
wraising2 = Rayleigh(2.068)
parents_wraising = [p]
CPD_wraising = CategoricalCPD(:h, name.(parents_wraising), [2], [wraising1, wraising2])
h = StdNode(CPD_wraising, parents_wraising)


parents_simduration = [t]
duration1 = NamedCategorical([:day1, :day10, :day100], [1.0, 0.0, 0.0])
duration2 = NamedCategorical([:day1, :day10, :day100], [0.0, 1.0, 0.0])
duration3 = NamedCategorical([:day1, :day10, :day100], [0.0, 0.0, 1.0])
CPDduration = CategoricalCPD(:k, name.(parents_simduration), [3], [duration1, duration2, duration3])
k = StdNode(CPDduration, parents_simduration)

wraising1 = Rayleigh(0.5)
wraising2 = Rayleigh(2.5)
wraising3 = Rayleigh(3.5)
parents_m = [t]
CPD_m = CategoricalCPD(:m, name.(parents_m), [3], [wraising1, wraising2, wraising3])
m = StdNode(CPD_m, parents_m)

par = [t, k, m, w, h]
