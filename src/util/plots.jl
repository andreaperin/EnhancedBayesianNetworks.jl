function gplot(net::AbstractNetwork;
    title="",
    title_color="black",
    title_size=4.0,
    font_family="Helvetica",
    nodelabelc="black",
    nodelabelsize=1.0,
    NODELABELSIZE=4.0,
    nodelabeldist=0.0,
    nodelabelangleoffset=π / 4.0,
    edgestrokec="black",
    edgelinewidth=1.0,
    EDGELINEWIDTH=1.0 / sqrt(length(net.nodes)),
    nodesizefactor=0.3,
    nodestrokec=nothing,
    nodestrokelw=0.0,
    arrowlengthfrac=0.1,
    arrowangleoffset=π / 9,
    background_color=nothing,
    plot_size=(15cm, 15cm),
    leftpad=0mm,
    rightpad=0mm,
    toppad=0mm,
    bottompad=0mm
)
    ## Title
    title_offset = isempty(title) ? 0 : 0.1 * title_size / 4 #Fix title offset
    title = Compose.text(0, -1.2 - title_offset / 2, title, hcenter, vcenter)
    ## Plot dim
    plot_area = (-1.2, -1.2 - title_offset, +2.4, +2.4 + title_offset)
    max_nodestrokelw = maximum(nodestrokelw)
    if max_nodestrokelw > 0.0
        max_nodestrokelw = EDGELINEWIDTH / max_nodestrokelw
        nodestrokelw *= max_nodestrokelw
    end
    max_edgelinewidth = EDGELINEWIDTH / maximum(edgelinewidth)
    edgelinewidth *= max_edgelinewidth

    node_list = _order_node(net.nodes)
    pos = _get_position(node_list)
    locs_x = map(p -> p[1], pos)
    locs_y = map(p -> p[2], pos)
    min_x, max_x = extrema(locs_x)
    min_y, max_y = extrema(locs_y)
    function scaler(z, a, b)
        if (a - b) == 0.0
            return 0.5
        else
            return 2.0 * ((z - a) / (b - a)) - 1.0
        end
    end
    map!(z -> scaler(z, min_x, max_x), locs_x, locs_x)
    map!(z -> scaler(z, min_y, max_y), locs_y, locs_y)
    nodecircle = fill(0.4 * 2.4, length(locs_x))
    nodesize = map(n -> length(String(n.name)), node_list)
    nodesize = normalize(nodesize) .* nodesizefactor
    for i = 1:length(locs_x)
        nodecircle[i] *= nodesize[i]
    end
    nodes = circle(locs_x, locs_y, nodecircle)

    nodelabel = map(n -> n.name, node_list)
    if !isnothing(nodelabel)
        text_locs_x = deepcopy(locs_x)
        text_locs_y = deepcopy(locs_y)
        texts = Compose.text(text_locs_x .+ nodesize .* (nodelabeldist * cos(nodelabelangleoffset)),
            text_locs_y .- nodesize .* (nodelabeldist * sin(nodelabelangleoffset)),
            map(string, nodelabel), [hcenter], [vcenter])
    end
    max_nodelabelsize = NODELABELSIZE / maximum(nodelabelsize)
    nodelabelsize *= max_nodelabelsize

    edges_list = _get_edges(get_adj_matrix(node_list))
    lines, larrows = _build_straight_edges(edges_list, locs_x, locs_y, nodesize, arrowlengthfrac, arrowangleoffset)
    nodestrokelw = map(n -> isa(n, DiscreteNode) ? 0.0 : 0.0, node_list)

    colors = _node_color.(node_list)

    Compose.set_default_graphic_size(plot_size...)
    compose(
        context(units=UnitBox(plot_area...; leftpad, rightpad, toppad, bottompad)),
        compose(context(), title, fill(title_color), fontsize(title_size), Compose.font(font_family)),
        compose(context(), texts, fill(nodelabelc), fontsize(nodelabelsize), Compose.font(font_family)),
        compose(context(), nodes, fill(colors), Compose.stroke(nodestrokec), linewidth(nodestrokelw)),
        # compose(context(), edgetexts, fill(edgelabelc), fontsize(edgelabelsize)),
        compose(context(), larrows, fill(edgestrokec)),
        # compose(context(), carrows, fill(edgestrokec)),
        compose(context(), lines, Compose.stroke(edgestrokec), linewidth(edgelinewidth)),
        # compose(context(), curves, stroke(edgestrokec), linewidth(edgelinewidth)),
        compose(context(units=UnitBox(plot_area...)), rectangle(plot_area...), fill(background_color))
    )
end

function _build_straight_edges(edge_list, locs_x, locs_y, nodesize, arrowlengthfrac, arrowangleoffset)
    if arrowlengthfrac > 0.0
        lines_cord, arrows_cord = _graphline(edge_list, locs_x, locs_y, nodesize, arrowlengthfrac, arrowangleoffset)
        lines = line(lines_cord)
        larrows = polygon(arrows_cord)
    else
        lines_cord = _graphline(edge_list, locs_x, locs_y, nodesize)
        lines = line(lines_cord)
        larrows = nothing
    end
    return lines, larrows
end

function _graphline(edge_list, locs_x, locs_y, nodesize::Vector{T}) where {T<:Real}
    num_edges = length(edge_list)
    lines = Array{Vector{Tuple{Float64,Float64}}}(undef, num_edges)
    for (e_idx, e) in enumerate(edge_list)
        i = e[1]
        j = e[2]
        Δx = locs_x[j] - locs_x[i]
        Δy = locs_y[j] - locs_y[i]
        θ = atan(Δy, Δx)
        startx = locs_x[i] + nodesize[i] * cos(θ)
        starty = locs_y[i] + nodesize[i] * sin(θ)
        endx = locs_x[j] + nodesize[j] * cos(θ + π)
        endy = locs_y[j] + nodesize[j] * sin(θ + π)
        lines[e_idx] = [(startx, starty), (endx, endy)]
    end
    lines
end

function _graphline(edge_list, locs_x, locs_y, nodesize::Vector{T}, arrowlength, angleoffset) where {T<:Real}
    num_edges = length(edge_list)
    lines = Array{Vector{Tuple{Float64,Float64}}}(undef, num_edges)
    arrows = Array{Vector{Tuple{Float64,Float64}}}(undef, num_edges)
    for (e_idx, e) in enumerate(edge_list)
        i = e[1]
        j = e[2]
        Δx = locs_x[j] - locs_x[i]
        Δy = locs_y[j] - locs_y[i]
        θ = atan(Δy, Δx)
        startx = locs_x[i] + nodesize[i] * cos(θ)
        starty = locs_y[i] + nodesize[i] * sin(θ)
        endx = locs_x[j] + nodesize[j] * cos(θ + π)
        endy = locs_y[j] + nodesize[j] * sin(θ + π)
        arr1, arr2 = _arrowcoords(θ, endx, endy, arrowlength, angleoffset)
        endx0, endy0 = _midpoint(arr1, arr2)
        e_idx2 = findfirst(==((j, i)), edge_list)
        if !isnothing(e_idx2) && e_idx2 < e_idx
            startx, starty = _midpoint(arrows[e_idx2][[1, 3]]...)
            lines[e_idx2][1] = (endx0, endy0)
        end
        lines[e_idx] = [(startx, starty), (endx0, endy0)]
        arrows[e_idx] = [arr1, (endx, endy), arr2]
    end
    lines, arrows
end

function _node_color(n::AbstractNode)
    if isa(n, FunctionalNode)
        if isa(n, DiscreteNode)
            return "lightsalmon"
        else
            return "red1"
        end
    elseif isa(n, ContinuousNode)
        if EnhancedBayesianNetworks._is_imprecise(n)
            return "cyan1"
        else
            return "paleturquoise"
        end
    elseif isa(n, DiscreteNode)
        if EnhancedBayesianNetworks._is_imprecise(n)
            return "green1"
        else
            return "palegreen"
        end
    end
end

function _arrowcoords(θ, endx, endy, arrowlength, angleoffset=20.0 / 180.0 * π)
    arr1x = endx - arrowlength * cos(θ + angleoffset)
    arr1y = endy - arrowlength * sin(θ + angleoffset)
    arr2x = endx - arrowlength * cos(θ - angleoffset)
    arr2y = endy - arrowlength * sin(θ - angleoffset)
    return (arr1x, arr1y), (arr2x, arr2y)
end

function _midpoint(pt1, pt2)
    x = (pt1[1] + pt2[1]) / 2
    y = (pt1[2] + pt2[2]) / 2
    return x, y
end

# ```
# List of available methods:

# `:spectral`, `:sfdp`, `:circular`, `:shell`, `:stress`, `:spring`, `:tree`, `:buchheim`, `:arcdiagram` or `:chorddiagram`

# ```

# function plot(bn::AbstractNetwork, layout=:spring, nodesize=0.1, fontsize=13)
#     if length(bn.nodes) > 1
#         graphplot(
#             bn.dag,
#             names=[i.name for i in bn.nodes],
#             # nodesize=map(x -> isa(x, ContinuousNode) ? Float64(0.2) : Float64(0.1), bn.nodes),
#             method=layout,
#             nodesize=nodesize,
#             fontsize=fontsize,
#             node_shape=map(x -> isa(x, ContinuousNode) ? :circle : :rect, bn.nodes),
#             markercolor=_marker_color.(bn.nodes),
#             linecolor=:darkgrey,
#         )
#     elseif length(bn.nodes) == 1
#         @warn ("Network is collapsed to a single node $(bn.nodes[1].name). Its probability table will be shown instead")
#         if isa(bn.nodes[1], DiscreteNode)
#             @show bn.nodes[1].states
#         else
#             @show bn.nodes[1].distribution
#         end
#     end
# end

# function _marker_color(node::AbstractNode)
#     if isa(node, FunctionalNode)
#         mc = "lightblue"

#     elseif isa(node, RootNode)
#         if isa(node, ContinuousNode)
#             if isa(node.distribution, UnivariateDistribution)
#                 mc = "orange"
#             else
#                 mc = "red"
#             end

#         elseif isa(node, DiscreteNode)
#             if isa(collect(values(node.states))[1], Real)
#                 mc = "orange"
#             else
#                 mc = "red"
#             end
#         end

#     elseif isa(node, ChildNode)
#         if isa(node, ContinuousNode)
#             if isa(collect(values(node.distribution))[1], UnivariateDistribution)
#                 mc = "orange"
#             else
#                 mc = "red"
#             end
#         elseif isa(node, DiscreteNode)
#             if isa(collect(values(collect(values(node.states))[1]))[1], Real)
#                 mc = "orange"
#             else
#                 mc = "red"
#             end
#         end
#     end
#     return mc
# end