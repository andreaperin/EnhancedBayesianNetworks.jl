struct ConditionalProbabilityTable
    data::DataFrame
    function ConditionalProbabilityTable(names::Vector{Symbol})
        colnames = [names..., :Π]
        data = DataFrame([name => [] for name in colnames])
        return new(data)
    end
end

function Base.setindex!(cpt::ConditionalProbabilityTable, value, key...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    cp = subset(cpt.data, selector, view=true)
    if isempty(cp)
        push!(cpt.data, (key..., Π=value))
    else
        @assert size(cp, 1) == 1
        cp.Π[1] = value
    end
    return nothing
end


function Base.getindex(cpt::ConditionalProbabilityTable, key...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    return subset(cpt.data, selector).Π[1]
end

x = ConditionalProbabilityTable([:a, :b])

x[:a=>:on, :b=>:off] = Normal()
x[:a=>:on, :b=>:on] = Exponential()

println("Distribution: $(x[:a => :on, :b => :on])")