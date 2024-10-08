using EnhancedBayesianNetworks
using Plots


v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
t = DiscreteStandardNode(:T, [v], Dict(
    [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
    [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
)

l = DiscreteStandardNode(:L, [s], Dict(
    [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
    [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
)

b = DiscreteStandardNode(:B, [s], Dict(
    [:yesS] => Dict(:yesB => 0.6, :noB => 0.4),
    [:noS] => Dict(:yesB => 0.3, :noB => 0.7))
)

e = DiscreteStandardNode(:E, [t, l], Dict(
    [:yesT, :yesL] => Dict(:yesE => 1, :noE => 0),
    [:yesT, :noL] => Dict(:yesE => 1, :noE => 0),
    [:noT, :yesL] => Dict(:yesE => 1, :noE => 0),
    [:noT, :noL] => Dict(:yesE => 0, :noE => 01))
)

d = DiscreteStandardNode(:D, [b, e], Dict(
    [:yesB, :yesE] => Dict(:yesD => 0.9, :noD => 0.1),
    [:yesB, :noE] => Dict(:yesD => 0.8, :noD => 0.2),
    [:noB, :yesE] => Dict(:yesD => 0.7, :noD => 0.3),
    [:noB, :noE] => Dict(:yesD => 0.1, :noD => 0.9))
)

x = DiscreteStandardNode(:X, [e], Dict(
    [:yesE] => Dict(:yesX => 0.98, :noX => 0.02),
    [:noE] => Dict(:yesX => 0.05, :noX => 0.95))
)

N = [v, s, t, l, b, e, d, x]
bn = BayesianNetwork(N)
gr();
nodesize = 0.1
fontsize = 18
EnhancedBayesianNetworks.plot(bn, :tree, nodesize, fontsize)
Plots.savefig("/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Conferences/2023_ESREL/ExtendedAbstract-Template/imgs/china.png")

evidence = Evidence()
factors = map(n -> Factor(bn, n.name, evidence), bn.nodes)
