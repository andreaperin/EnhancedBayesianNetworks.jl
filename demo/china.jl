using EnhancedBayesianNetworks
using Plots


v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
t = DiscreteStandardNode(:T, [v], OrderedDict(
    [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
    [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
)

l = DiscreteStandardNode(:L, [s], OrderedDict(
    [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
    [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
)

b = DiscreteStandardNode(:B, [s], OrderedDict(
    [:yesS] => Dict(:yesB => 0.6, :noB => 0.4),
    [:noS] => Dict(:yesB => 0.3, :noB => 0.7))
)

e = DiscreteStandardNode(:E, [t, l], OrderedDict(
    [:yesT, :yesL] => Dict(:yesE => 1, :noE => 0),
    [:yesT, :noL] => Dict(:yesE => 1, :noE => 0),
    [:noT, :yesL] => Dict(:yesE => 1, :noE => 0),
    [:noT, :noL] => Dict(:yesE => 0, :noE => 01))
)

d = DiscreteStandardNode(:D, [b, e], OrderedDict(
    [:yesB, :yesE] => Dict(:yesD => 0.9, :noD => 0.1),
    [:yesB, :noE] => Dict(:yesD => 0.8, :noD => 0.2),
    [:noB, :yesE] => Dict(:yesD => 0.7, :noD => 0.3),
    [:noB, :noE] => Dict(:yesD => 0.1, :noD => 0.9))
)

x = DiscreteStandardNode(:X, [e], OrderedDict(
    [:yesE] => Dict(:yesX => 0.98, :noX => 0.02),
    [:noE] => Dict(:yesX => 0.05, :noX => 0.95))
)

N = [v, s, t, l, b, e, d, x]
bn = BayesianNetwork(N)

query = :V
evidence = Evidence()
c = infer(bn, query, evidence)

# elimination_oreder = [:D, :B, :L, :S, :T]
# factors = map(n -> Factor(bn, n.name, evidence), bn.nodes)
# h = elimination_oreder[3]
# contain_h = filter(ϕ -> h ∈ ϕ, factors)
