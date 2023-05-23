@testset "linear" begin
    X = Float64[1;2;3;4]
    y = Float64[1;2;3;4]
    @test linear(X, y) == 30
end

@testset "polynomial" begin
    X = Float64[1;2;3;4]
    y = Float64[1;2;3;4]
    @test polynomial(X, y, 0.0, 2) == 900
end

@testset "rbf" begin
    X = Float64[1;2;3;4]
    y = Float64[1;2;3;4]
    @test rbf(X, y, 1.0) == 1.0
end

@testset "kernel" begin
    X = Float64[1;2;3;4]
    y = Float64[1;2;3;4]
    answer = Dict(
        "linear" => Dict("answer" => 30.0, "kwargs" => Dict()),
        "polynomial" => Dict("answer" => 900.0, "kwargs" => Dict(:c => 0.0, :d => 2)),
        "rbf" => Dict("answer" => 1.0, "kwargs" => Dict(:γ => 1.0))
    )
    for (key, value) in answer
        @test kernel_find(SVC(;kernel=key, value["kwargs"]...), X, y) == value["answer"]
    end
end