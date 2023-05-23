using LinearAlgebra
function linear(X_i::Vector{Float64}, X_j::Vector{Float64})::Float64
    return X_i' * X_j
end

function polynomial(X_i::Vector{Float64}, X_j::Vector{Float64}, c::Float64, d::Int64)::Float64
    return (X_i' * X_j + c) ^ d
end


function rbf(X_i::Vector{Float64}, X_j::Vector{Float64}, γ::Float64)::Float64
    return exp(-γ * norm(X_i - X_j) ^ 2)
end
    
