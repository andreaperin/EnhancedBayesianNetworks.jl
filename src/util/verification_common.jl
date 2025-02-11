function _by_row(evidence::Dict{Symbol,Symbol})
    k = collect(keys(evidence))
    v = collect(values(evidence))
    return map((n, s) -> n => ByRow(x -> x == s), k, v)
end

function _verify_prob_column(cpt::DataFrame)
    if "Π" ∉ names(cpt)
        error("cpt must contain a column named :Π where probabilities are collected: $cpt")
    end
end