# reduce the dimension and then squeeze them out
Base.sum(ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}) = _reducedim(+, ϕ, dims)

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
    broadcast!(op, new_v, ϕ1.potential, temp)

    return Factor(new_dims, new_v, new_states_mapping)
end

*(ϕ1::Factor, ϕ2::Factor) = join(*, ϕ1, ϕ2)
