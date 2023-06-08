using EnhancedBayesianNetworks

a = DiscreteRootNode(:a, Dict(:a1 => 0.3, :a2 => 0.7))
b = DiscreteRootNode(:b, Dict(:b1 => 0.2, :b2 => 0.7, :b3 => 0.1))
c = DiscreteStandardNode(:c, [a, b], OrderedDict(
    [:a1, :b1] => Dict(:c1 => 0.1, :c2 => 0.9),
    [:a2, :b1] => Dict(:c1 => 0.4, :c2 => 0.6),
    [:a1, :b2] => Dict(:c1 => 0.2, :c2 => 0.8),
    [:a2, :b2] => Dict(:c1 => 0.0, :c2 => 1.0),
    [:a1, :b3] => Dict(:c1 => 1.0, :c2 => 0.0),
    [:a2, :b3] => Dict(:c1 => 0.0, :c2 => 1.0)
))

bn = BayesianNetwork([a, b, c])
evidence = Evidence(Dict(:c => :c1))
query = [:a, :b]


c = infer(bn, :b, Dict(:a => :a1, :c => :c2))