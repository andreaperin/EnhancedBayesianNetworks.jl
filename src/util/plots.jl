function plot(bn::Union{BayesianNetwork,EnhancedBayesianNetwork,ReducedBayesianNetwork}, layout=:tree, nodesize=0.1, fontsize=13)
    graphplot(
        bn.dag,
        names=[i.name for i in bn.nodes],
        # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), bn.nodes),
        method=layout,
        nodesize=nodesize,
        fontsize=fontsize,
        node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, bn.nodes),
        markercolor=_marker_color.(bn.nodes),
        linecolor=:darkgrey,
    )
end

function _marker_color(node::AbstractNode)
    if isa(node, FunctionalNode)
        mc = "lightgreen"
    elseif isa(node, StructuralReliabilityProblemNode)
        mc = "lightblue"
    else
        mc = "orange"
    end
    return mc
end

```
List of available methods:

`:spectral`, `:sfdp`, `:circular`, `:shell`, `:stress`, `:spring`, `:tree`, `:buchheim`, `:arcdiagram` or `:chorddiagram`

```
# function plot(bn::BayesianNetwork)
#     graphplot(
#         bn.dag,
#         names=[i.name for i in bn.nodes],
#         # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), bn.nodes),
#         font_size=14,
#         node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, bn.nodes),
#         markercolor=map(x -> isa(x, DiscreteFunctionalNode) ? "lightgreen" : "orange", bn.nodes),
#         linecolor=:darkgrey,
#     )
# end

# function plot(ebn::EnhancedBayesianNetwork)
#     graphplot(
#         ebn.dag,
#         names=[i.name for i in ebn.nodes],
#         # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), ebn.nodes),
#         font_size=14,
#         node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, ebn.nodes),
#         markercolor=map(x -> isa(x, DiscreteFunctionalNode) ? "lightgreen" : "orange", ebn.nodes),
#         linecolor=:darkgrey,
#     )
# end

# function plot(ebn::ReducedBayesianNetwork)
#     graphplot(
#         ebn.dag,
#         names=[i.name for i in ebn.nodes],
#         # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), ebn.nodes),
#         font_size=14,
#         node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, ebn.nodes),
#         markercolor=map(x -> isa(x, DiscreteFunctionalNode) ? "lightgreen" : "orange", ebn.nodes),
#         linecolor=:darkgrey,
#     )
# end