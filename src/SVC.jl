using JuMP
using GLPK
using Ipopt
using MathOptInterface
using Match
include("kernel.jl")
const MOI = MathOptInterface
mutable struct SVC
    kernel::String
    W::Vector{Float64}
    b::Float64
    c::Float64
    regularizer::Float64
    d::Int64
    γ::Float64
    cost_values::Vector{Float64}
    scores::Vector{Float64}
    support_vectors::Vector{Float64}
    support_vectors_labels::Vector{Float64}
    function SVC(; kernel::String="linear", regularizer::Float64=1.0, c::Float64=0.0, d::Int64=0, γ::Float64=0.0)
        W::Vector{Float64} = []
        b::Float64 = 0.0
        cost_values::Vector{Float64} = []
        scores::Vector{Float64} = []
        support_vectors::Vector{Float64} = []
        support_vectors_labels::Vector{Float64} = []
        new(kernel, W, b, c, regularizer, d, γ, cost_values, scores, support_vectors, support_vectors_labels)
    end
end




function fit!(svc::SVC, X::Matrix{Float64}, y::Vector{Float64})
    m, n = size(X)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    @variable(model, 0 <= α[i=1:m] <= svc.regularizer)
    @constraint(model, sum(α[i] * y[i] for i = 1:m) == 0)
    @objective(model,
        Max,
        sum(α[i] for i = 1:m) - 0.5 * sum(sum(α[i] * α[j] * y[i] * y[j] * kernel_find(svc, X[i, :], X[j, :]) for i = 1:m) for j = 1:m)
    )
    function objective_callback(alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint)
        if iter_count != 0
            α_optimal = [callback_value(model, α[i]) for i = 1:m]
            svc.W = sum(α_optimal[i] * y[i] * X[i, :] for i = 1:m)
            support_idx = findall(α -> α > 0, α_optimal)
            support_vectors = X[support_idx, :]
            support_labels = y[support_idx]
            svc.b = support_labels[begin] - svc.W' * support_vectors[begin, :]
            push!(svc.cost_values, obj_value)
            push!(svc.scores, accuracy(y, predict(svc, X)))
        end
        return iter_count < 100
    end
    MOI.set(model, Ipopt.CallbackFunction(), objective_callback)
   
    optimize!(model)

    return svc
end

function predict(svc::SVC, X::Matrix{Float64})::Vector{Float64}
    return sign.(svc.W' * X' .+ svc.b) |> vec
end

function accuracy(y_true::Vector{Float64}, y_pred::Vector{Float64})
    correct_predictions = sum(y_true .== y_pred)
    return correct_predictions / length(y_true)
end

function kernel_find(svc::SVC, X_i::Vector{Float64}, X_j::Vector{Float64})::Float64
    return @match svc.kernel begin
        "linear" => linear(X_i, X_j)
        "polynomial" => polynomial(X_i, X_j, svc.c, svc.d)
        "rbf" => rbf(X_i, X_j, svc.γ)
    end
end