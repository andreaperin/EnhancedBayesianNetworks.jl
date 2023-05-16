function verify_intervals(intervals::Vector{Vector{Float64}})
    for i in intervals
        i[1] > i[2] && error("malformed interval")
        for j in intervals
            if i != j
                minimum(maximum.([i, j])) > maximum(minimum.([i, j])) && error("overlapping intervals")
            end
        end
    end
    deleteat!(sort(minimum.(intervals)), 1) != deleteat!(sort(maximum.(intervals)), length(intervals)) && error("non continuous range of intervals")
end