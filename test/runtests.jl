using SVM
using Test
using MLJ: make_blobs
using Plots
include("kernel.jl")

@testset "main" begin
    n_samples = 100
    n_features = 2
    n_centers = 2
    cluster_std = 1.0
    centers = 2
    
    X, y = make_blobs(n_samples, n_features; centers=centers, cluster_std=cluster_std, as_table=false, center_box=(-10. => 10.))
    y = map(x -> x == 1 ? -1.0 : 1.0, y)
    model = SVC()
    (;W, b, support_vectors, cost_values, scores) = fit!(model, X, y)
    plot(scores)
    savefig("cost_values.png")
    # scatter(X[:, 1], X[:, 2], color=convert(Vector{Int64}, y))
    # scatter!(support_vecteur[:, 1], [support_vecteur[:, 2]], color=:red)
    # x_hyperplane = range(-10, 10, length=100) |> collect
    # y_hyperplane = (-b .- w[1] * x_hyperplane) / w[2]
    # y_support_vecteur_plus = (-b .- w[1] * x_hyperplane .+ 1) / w[2]
    # y_support_vecteur_moins = (-b .- w[1] * x_hyperplane .- 1) / w[2]
    # plot!(x_hyperplane, y_hyperplane, color=:green)
    # plot!(x_hyperplane, y_support_vecteur_plus, color=:blue)
    # plot!(x_hyperplane, y_support_vecteur_moins, color=:blue)
    # max = 10
    # xlims!(-max, max)
    # ylims!(-max, max)
    # savefig("test.png")

end

"Done."