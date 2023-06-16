## Methods
Base.size(ϕ::Factor) = size(ϕ.potential)
Base.size(ϕ::Factor, dim::Symbol) = size(ϕ.potential, indexin(dim, ϕ))
Base.names(ϕ::Factor) = ϕ.dimensions
Base.in(dim::Symbol, ϕ::Factor) = dim in names(ϕ)
Base.indexin(dim::Symbol, ϕ::Factor) = findnext(isequal(dim), ϕ.dimensions, 1)
Base.indexin(dims::Vector{Symbol}, ϕ::Factor) = indexin(dims, names(ϕ))
Base.ndims(ϕ::Factor) = ndims(ϕ.potential)
Base.length(ϕ::Factor) = length(ϕ.potential)


_reddim(op, ϕ::Factor, inds::Tuple, ::Nothing) =
    dropdims(reduce(op, ϕ.potential, dims=inds), dims=inds)
_reddim(op, ϕ::Factor, inds::Tuple, v0) =
    dropdims(reducedim(op, ϕ.potential, inds, v0), dims=inds)

function reducedim(op, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, v0=nothing)
    dims = wrap(dims)
    _check_dims_valid(dims, ϕ)
    # needs to be a tuple for squeeze
    inds = (indexin(dims, ϕ)...,)

    dims_new = deepcopy(ϕ.dimensions)
    deleteat!(dims_new, inds)

    v_new = _reddim(op, ϕ, inds, v0)
    states_new = Dict(k => v for (k, v) in ϕ.states_mapping if k ∉ dims)

    ϕ = Factor(dims_new, v_new, states_new)
    return ϕ
end

function reducedim!(op, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, v0=nothing)
    dims = wrap(dims)
    _check_dims_valid(dims, ϕ)
    # needs to be a tuple for squeeze
    inds = (indexin(dims, ϕ)...,)
    deleteat!(ϕ.dimensions, inds)
    ϕ.potential = _reddim(op, ϕ, inds, v0)
    ϕ.states_mapping = Dict(k => v for (k, v) in ϕ.states_mapping if k ∉ dims)
    return ϕ
end

@inline function Base.permutedims!(ϕ::Factor, perm)
    ϕ.potential = permutedims(ϕ.potential, perm)
    ϕ.dimensions = ϕ.dimensions[perm]
    return ϕ
end

Base.permutedims(ϕ::Factor, perm) = permutedims!(deepcopy(ϕ), perm)


Base.broadcast(f, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, values) = broadcast!(f, deepcopy(ϕ), dims, values)

function Base.broadcast!(f, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, values)
    if isa(dims, Symbol)
        dims = [dims]
        values = [values]
    end
    _ckeck_dims_unique(dims)
    _check_dims_valid(dims, ϕ)

    (length(dims) != length(values)) &&
        error("Number of dimensions does not " * "match number of values to broadcast")

    inds = indexin(dims, ϕ)

    reshape_dims = ones(Int, ndims(ϕ))
    new_values = Vector{Array{Float64}}(undef, length(values))

    for (i, val) in enumerate(values)
        if isa(val, Vector{Float64})
            # reshape to the proper dimension
            dim_loc = inds[i]
            @inbounds reshape_dims[dim_loc] = length(val)
            new_values[i] = reshape(val, reshape_dims...)
            @inbounds reshape_dims[dim_loc] = 1
        elseif isa(val, Float64)
            new_values[i] = [val]
        else
            throw(TypeError(:broadcast!, "Invalid broadcast value",
                Union{Float64,Vector{Float64}}, val))
        end
    end

    broadcast!(f, ϕ.potential, ϕ.potential, new_values...)

    return ϕ
end

function Base.getindex(ϕ::Factor, e::Evidence)
    inds = _translate_index(ϕ, e)
    keep = inds .== Colon()
    new_dims = ϕ.dimensions[keep]
    @inbounds new_p = ϕ.potential[inds...]

    if ndims(new_p) == 0
        new_p = dropdims([new_p], dims=1)
    end

    states_mapping_dict = filter(((k, v),) -> k ∈ new_dims, ϕ.states_mapping)
    return Factor(new_dims, new_p, states_mapping_dict)
end

Base.getindex(ϕ::Factor, pair::Pair{Symbol}...) = Base.getindex(ϕ, Evidence(pair))
