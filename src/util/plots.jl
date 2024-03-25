```
List of available methods:

`:spectral`, `:sfdp`, `:circular`, `:shell`, `:stress`, `:spring`, `:tree`, `:buchheim`, `:arcdiagram` or `:chorddiagram`

```

function plot(bn::Union{BayesianNetwork,EnhancedBayesianNetwork}, layout=:spring, nodesize=0.1, fontsize=13)
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
    else
        mc = "orange"
    end
    return mc
end