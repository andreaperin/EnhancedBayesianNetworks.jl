function _by_row(evidence::Dict{Symbol,Symbol})
    k = collect(keys(evidence))
    v = collect(values(evidence))
    return map((n, s) -> n => ByRow(x -> x == s), k, v)
end

function _verify_prob_column(cpt::DataFrame)
    if "Prob" ∉ names(cpt)
        error("cpt must contain a column named :Prob where probabilities are collected: $cpt")
    end
end