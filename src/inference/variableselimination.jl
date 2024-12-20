function infer(inf::PreciseInferenceState)
    bn = inf.bn
    nodes = bn.nodes
    query = inf.query
    evidence = inf.evidence
    factors = map(n -> Factor(bn, n.name, evidence), nodes)
    # successively remove the hidden nodes
    δ = [x[1] for x in _minimal_increase_in_complexity(factors, bn.name_to_index)]
    δ = deleteat!(δ, findall(x -> x ∈ vcat(query, collect(keys(evidence))), δ))

    while !isempty(δ)
        h = first(δ)
        contain_h = filter(ϕ -> h in ϕ, factors)
        if !isempty(contain_h)
            factors = setdiff(factors, contain_h)
            τ_h = sum(reduce((*), contain_h), h)
            push!(factors, τ_h)
        end
        δ = [x[1] for x in _minimal_increase_in_complexity(factors, bn.name_to_index)]
        δ = deleteat!(δ, findall(x -> x ∈ vcat(query, collect(keys(evidence))), δ))
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

    bns = map(anc -> BayesianNetwork(anc), all_nodes_combination)

    r = map(bn -> infer(bn, query, evidence), bns)

    res = stack(map(r -> r.potential, r))

    a = minimum(res; dims=dims)
    b = maximum(res; dims=dims)

    potential = map((a, b) -> [a, b], a, b)
    potential = reshape(potential, size(r[1].potential))
    return Factor(r[1].dimensions, potential, r[1].states_mapping)
end


function _minimal_increase_in_complexity(factors::Vector{Factor}, name_to_index::Dict{Symbol,Int64})
    g = _moral_graph_from_dimensions([i.dimensions for i in factors], name_to_index)
    res = Tuple[]
    filter!(x -> !isempty(x.dimensions), factors)
    for factor in factors
        node = first(factor.dimensions)
        ψ = filter(x -> node ∈ x.dimensions, factors)
        n_e = mapreduce(x -> length(x.dimensions) - 1, +, ψ)
        fadjlist = deepcopy(g.fadjlist)
        k = filter(x -> name_to_index[node] ∈ x, fadjlist)
        deleteat!(fadjlist, findall(x -> x ∈ k, fadjlist))

        new_links = filter!(x -> x != name_to_index[node], collect(Iterators.flatten(k)))
        collection = collect(Iterators.product(new_links, new_links))
        collection = map(t -> [t...], collection)
        utri = triu!(trues(size(collection)))
        for i in eachindex(utri)
            rowi, coli = fldmod1(i, size(collection, 2))
            if utri[rowi, coli] == true && collection[rowi, coli][1] != collection[rowi, coli][2]
                if collection[rowi, coli] ∉ fadjlist && [collection[rowi, coli][2], collection[rowi, coli][1]] ∉ fadjlist
                    push!(fadjlist, collection[rowi, coli])
                end
            end
        end
        n_a = length(fadjlist) - (length(g.fadjlist) - n_e)
        push!(res, (node, n_a / n_e))
    end
    return sort(res, by=x -> x[2])
end

function _moral_graph_from_dimensions(dimensions::Vector{Vector{Symbol}}, name_to_index::Dict{Symbol,Int64})
    list = Vector{Vector{Int64}}()
    for dim in dimensions
        if length(dim) != 1
            collection = collect(Iterators.product(dim, dim))
            collection = map(t -> [t...], collection)
            collection = map(v -> [name_to_index[v[1]], name_to_index[v[2]]], collection)
            utri = triu!(trues(size(collection)))
            for i in eachindex(utri)
                rowi, coli = fldmod1(i, size(collection, 2))
                if utri[rowi, coli] == true && collection[rowi, coli][1] != collection[rowi, coli][2]
                    push!(list, collection[rowi, coli])
                end
            end
        end
    end
    SimpleGraph(length(name_to_index), list)
end

infer(bn::BayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence=Evidence()) =
    infer(PreciseInferenceState(bn, query, evidence))

infer(cn::CredalNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence=Evidence()) =
    infer(ImpreciseInferenceState(cn, query, evidence))