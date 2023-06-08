# reduce the dimension and then squeeze them out
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
    states_mapping_new = Dict(k => v for (k, v) in ϕ.states_mapping if k ∉ dims)
    v_new = _reddim(op, ϕ, inds, v0)
    ϕ = Factor(dims_new, v_new, states_mapping_new)
    return ϕ
end

function reducedim!(op, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, v0=nothing)
    dims = wrap(dims)
    _check_dims_valid(dims, ϕ)
    # needs to be a tuple for squeeze
    inds = (indexin(dims, ϕ)...,)
    deleteat!(ϕ.dimensions, inds)
    ϕ.potential = _reddim(op, ϕ, inds, v0)
    return ϕ
end

Base.sum(ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}) = reducedim(+, ϕ, dims)



@inline function Base.permutedims!(ϕ::Factor, perm)
    ϕ.potential = permutedims(ϕ.potential, perm)
    ϕ.dimensions = ϕ.dimensions[perm]
    return ϕ
end

Base.permutedims(ϕ::Factor, perm) = permutedims!(deepcopy(ϕ), perm)

function Base.broadcast!(f, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, values)
    if isa(dims, Symbol)
        dims = [dims]
        values = [values]
    end

    _ckeck_dims_unique(dims)
    _check_dims_valid(dims, ϕ)

    (length(dims) != length(values)) &&
        error("Number of dimensions does not " * "match number of values to broadcast")

    # broadcast will check if the dimensions of each value are valid

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


function Base.join(op, ϕ1::Factor, ϕ2::Factor)
    if length(ϕ1) < length(ϕ2)
        ϕ2, ϕ1 = ϕ1, ϕ2
    end

    common = intersect(ϕ1.dimensions, ϕ2.dimensions)
    index_common1 = indexin(common, ϕ1.dimensions)
    index_common2 = indexin(common, ϕ2.dimensions)

    if [size(ϕ1)[index_common1]...] != [size(ϕ2)[index_common2]...]
        error("Common dimensions must have same size")
    end

    # the first dimensions are all from ϕ1
    new_dims = union(ϕ1.dimensions, ϕ2.dimensions)
    new_states_mapping = merge(ϕ1.states_mapping, ϕ2.states_mapping)

    if ndims(ϕ2.potential) != 0
        # permute the common dimensions in ϕ2 to the beginning,
        #  in the order that they appear in ϕ1 (and therefore new_dims)            
        unique1 = setdiff(ϕ1.dimensions, common)
        unique2 = setdiff(ϕ2.dimensions, common)
        # these will also be the same indices for new_dims
        index_unique1 = indexin(unique1, ϕ1.dimensions)
        index_unique2 = indexin(unique2, ϕ2.dimensions)
        perm = vcat(index_common2, index_unique2)
        temp = permutedims(ϕ2.potential, perm)
        # reshape by lining up the common dims in ϕ2 with those in ϕ1
        size_unique2 = size(ϕ2)[index_unique2]
        # set those dims to have dimension 1 for data in ϕ2
        reshape_lengths = vcat(size(ϕ1)..., size_unique2...)
        new_v = Array{Float64}(undef, reshape_lengths...)
        reshape_lengths[index_unique1] .= 1
        temp = reshape(temp, (reshape_lengths...,))
    else
        new_v = similar(ϕ1.potential)
        temp = ϕ2.potential
    end
    # ndims(ϕ1) == 0 implies ndims(ϕ2) == 0
    if ndims(ϕ1.potential) == 0
        new_v = dropdims([op(ϕ1.potential[1], temp[1])], dims=1)
    else
        broadcast!(op, new_v, ϕ1.potential, temp)
    end
    return Factor(new_dims, new_v, new_states_mapping)
end

*(ϕ1::Factor, ϕ2::Factor) = join(*, ϕ1, ϕ2)
