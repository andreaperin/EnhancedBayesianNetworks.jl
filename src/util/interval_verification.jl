function verify_intervals(intervals::Vector{Vector{Float64}})
    if sort(intervals) != intervals
        error("interval values $intervals are not sorted")
    end
end
