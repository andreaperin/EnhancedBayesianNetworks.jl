function infer(inf::PreciseInferenceState)
    bn = inf.bn
    nodes = bn.nodes
    query = inf.query
    evidence = inf.evidence
    factors = map(n -> Factor(bn, n.name, evidence), nodes)
    # successively remove the hidden nodes
    δ = [x[1] for x in _order_with_minimal_increase_in_complexity(factors, bn.topology_dict)]
    δ = deleteat!(δ, findall(x -> x ∈ vcat(query, collect(keys(evidence))), δ))
    list = []
    while !isempty(δ)
        h = first(δ)
        push!(list, h)
        contain_h = filter(ϕ -> h ∈ ϕ, factors)
        if !isempty(contain_h)
            factors = setdiff(factors, contain_h)
            τ_h = sum(reduce((*), contain_h), h)
            push!(factors, τ_h)
        end
        δ = [x[1] for x in _order_with_minimal_increase_in_complexity(factors, bn.topology_dict)]
        δ = deleteat!(δ, findall(x -> x ∈ vcat(query, collect(keys(evidence)), list), δ))
    end
    ϕ = reduce((*), factors)
    tot = sum(abs, ϕ.potential)
    ϕ.potential ./= tot
    return ϕ
end

function infer(inf::ImpreciseInferenceState)
    cn = inf.cn
    nodes = cn.nodes
    query = inf.query
    evidence = inf.evidence

    dims = length(query) + 1
    all_nodes = map(node -> _extreme_points(node), nodes)
    all_nodes_combination = vec(collect(Iterators.product(all_nodes...)))
    all_nodes_combination = map(t -> [t...], all_nodes_combination)

    bns = map(anc -> BayesianNetwork(anc, cn.topology_dict, cn.adj_matrix), all_nodes_combination)

    r = map(bn -> infer(bn, query, evidence), bns)

    res = stack(map(r -> r.potential, r))

    a = minimum(res; dims=dims)
    b = maximum(res; dims=dims)

    potential = map((a, b) -> [a, b], a, b)
    potential = reshape(potential, size(r[1].potential))
    return Factor(r[1].dimensions, potential, r[1].states_mapping)
end

function _order_with_minimal_increase_in_complexity(factors::Vector{Factor}, topology_dict::Dict{Symbol,Int64})
    dimensions = map(f -> f.dimensions, factors)
    res = map(x -> (x, _n_added_edges(dimensions, topology_dict, topology_dict[x]) / _n_eliminated_edges(dimensions, topology_dict, topology_dict[x])), collect(keys(topology_dict)))
    return sort(res, by=x -> x[2])
end

function _n_eliminated_edges(dimensions::AbstractVector{Vector{Symbol}}, topology_dict::Dict{Symbol,Int}, index::Int)
    structure_adj_matrix = _structure_adj_matrix(dimensions, topology_dict)
    return length(structure_adj_matrix[index, :].nzind)
end

function _n_added_edges(dimensions::AbstractVector{Vector{Symbol}}, topology_dict::Dict{Symbol,Int}, index::Int)
    reverse_dict = Dict(value => key for (key, value) in topology_dict)
    node = reverse_dict[index]
    structure_adj_matrix = _structure_adj_matrix(dimensions, topology_dict)
    former_edges = length(structure_adj_matrix.nzval) - 2 * _n_eliminated_edges(dimensions, topology_dict, index)
    function _ridimensionalize(d::AbstractVector{Symbol})
        return filter(x -> x != node, d)
    end
    ## Adding the new connection among parents and children
    new_connection = filter(x -> node ∈ x, dimensions)
    new_dims = map(dim -> _ridimensionalize(dim), dimensions)
    if !isempty(new_connection)
        new_connection = mapreduce(x -> _ridimensionalize(x), vcat, new_connection)
        push!(new_dims, new_connection)
    end
    function _retopologyse(topo::Dict{Symbol,Int})
        new_dict = Dict{Symbol,Int}()
        for (k, v) in collect(topo)
            if k != node
                if v > index
                    new_dict[k] = v - 1
                else
                    new_dict[k] = v
                end
            end
        end
        return new_dict
    end
    new_topology_dict = _retopologyse(topology_dict)
    new_structure_adj_matrix = _structure_adj_matrix(new_dims, new_topology_dict)
    return Int((length(new_structure_adj_matrix.nzval) - former_edges) / 2)
end

function _structure_adj_matrix(dimensions::AbstractVector{Vector{Symbol}}, topology_dict::Dict{Symbol,Int})
    n = length(topology_dict)
    structure_adj_matrix = zeros(n, n)

    function _structure_link(dim::AbstractVector{Symbol})
        links = Vector{}()
        if length(dim) > 1
            collection = collect(Iterators.product(dim, dim))
            collection = map(t -> [t...], collection)
            collection = vec(map(v -> [topology_dict[v[1]], topology_dict[v[2]]], collection))
            filter!(c -> c[1] != c[2], collection)
            append!(links, collection)
        end
        return links
    end

    structural_links = unique!(mapreduce(dim -> _structure_link(dim), vcat, dimensions))
    for link in structural_links
        structure_adj_matrix[link[1], link[2]] = 1
    end
    return sparse(structure_adj_matrix)
end

infer(bn::BayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence=Evidence()) = infer(PreciseInferenceState(bn, query, evidence))

infer(cn::CredalNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence=Evidence()) = infer(ImpreciseInferenceState(cn, query, evidence))