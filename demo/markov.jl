using EnhancedBayesianNetworks
using Plots

x1 = DiscreteRootNode(:x1, Dict(:x1y => 0.5, :x1n => 0.5))
x2 = DiscreteRootNode(:x2, Dict(:x2y => 0.5, :x2n => 0.5))
x4 = DiscreteRootNode(:x4, Dict(:x4y => 0.5, :x4n => 0.5))
x8 = DiscreteRootNode(:x8, Dict(:x8y => 0.5, :x8n => 0.5))

x3_states = Dict(
    [:x1y] => Dict(:x3y => 0.5, :x3n => 0.5),
    [:x1n] => Dict(:x3y => 0.5, :x3n => 0.5)
)
x3 = DiscreteChildNode(:x3, [x1], x3_states)


x5_states = Dict(
    [:x2y] => Dict(:x5y => 0.5, :x5n => 0.5),
    [:x2n] => Dict(:x5y => 0.5, :x5n => 0.5)
)
x5 = DiscreteChildNode(:x5, [x2], x5_states)

x7_states = Dict(
    [:x4y] => Dict(:x7y => 0.5, :x7n => 0.5),
    [:x4n] => Dict(:x7y => 0.5, :x7n => 0.5)
)
x7 = DiscreteChildNode(:x7, [x4], x7_states)

x11_states = Dict(
    [:x8y] => Dict(:x11y => 0.5, :x11n => 0.5),
    [:x8n] => Dict(:x11y => 0.5, :x11n => 0.5)
)
x11 = DiscreteChildNode(:x11, [x8], x11_states)

x6_states = Dict(
    [:x4y, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
    [:x4y, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5),
    [:x4n, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
    [:x4n, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5)
)
x6 = DiscreteChildNode(:x6, [x4, x3], x6_states)

x6_states = Dict(
    [:x4y, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
    [:x4y, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5),
    [:x4n, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
    [:x4n, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5)
)
x6 = DiscreteChildNode(:x6, [x4, x3], x6_states)


x9_states = Dict(
    [:x6y, :x5y] => Dict(:x9y => 0.5, :x9n => 0.5),
    [:x6y, :x5n] => Dict(:x9y => 0.5, :x9n => 0.5),
    [:x6n, :x5y] => Dict(:x9y => 0.5, :x9n => 0.5),
    [:x6n, :x5n] => Dict(:x9y => 0.5, :x9n => 0.5)
)
x9 = DiscreteChildNode(:x9, [x6, x5], x9_states)

x10_states = Dict(
    [:x6y, :x8y] => Dict(:x10y => 0.5, :x10n => 0.5),
    [:x6y, :x8n] => Dict(:x10y => 0.5, :x10n => 0.5),
    [:x6n, :x8y] => Dict(:x10y => 0.5, :x10n => 0.5),
    [:x6n, :x8n] => Dict(:x10y => 0.5, :x10n => 0.5)
)
x10 = DiscreteChildNode(:x10, [x6, x8], x10_states)

x12_states = Dict(
    [:x9y] => Dict(:x12y => 0.5, :x12n => 0.5),
    [:x9n] => Dict(:x12y => 0.5, :x12n => 0.5)
)
x12 = DiscreteChildNode(:x12, [x9], x12_states)

x13_states = Dict(
    [:x10y] => Dict(:x13y => 0.5, :x13n => 0.5),
    [:x10n] => Dict(:x13y => 0.5, :x13n => 0.5)
)
x13 = DiscreteChildNode(:x13, [x10], x13_states)

nodes = [x1, x2, x4, x8, x5, x7, x11, x3, x6, x9, x10, x12, x13]

ebn = EnhancedBayesianNetwork(nodes)
EnhancedBayesianNetworks.plot(ebn)
