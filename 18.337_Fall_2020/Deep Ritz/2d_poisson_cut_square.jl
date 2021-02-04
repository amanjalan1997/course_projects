using Flux
using Flux: @epochs
using Flux: params
using Flux: throttle
using Plots
using Distributions
using DataFrames
using CSV
using StaticArrays

@inline phi(x) = max(x^3,0)
@inline f(x) = 1

#build a block
block_layer = SkipConnection(Chain(Dense(10,10, phi),Dense(10,10, phi)), +)

#construct network skeleton
M = Chain(Dense(2,10),
block_layer,
block_layer,
block_layer,
block_layer,
Dense(10,1))

#check number of params
params(M)

#inner loss function
function inner_loss(N, space)
    a = rand(Uniform(-1,1),2,N)
    loss = 0
    wt = 4*ones(N)/N
    l11 = [1, 0]
    l21 = [2, 0]
    l12 = [0, 1]
    l22 = [0, 2]

    @inbounds loss += ((0.5*((-1/12*M(a .+ space.*l21) .+ 2/3*M(a .+ space.*l11) .- 2/3*M(a .- space.*l11) .+ 1/12*M(a .- space.*l21))./space).^2) *wt)[1]
    @inbounds loss += ((0.5*((-1/12*M(a .+ space.*l22) .+ 2/3*M(a .+ space.*l12) .- 2/3*M(a .- space.*l12) .+ 1/12*M(a .- space.*l22))./space).^2) *wt)[1]
    @inbounds loss += (-1 .*M(a)*wt)[1]

    return loss
end

#boundary condition loss function
#add [0,1) as a boundary
function bc_loss(N, β)
    dist = Uniform(-1,1)
    a = hcat(vcat(rand(dist, 1,floor(Int32, N/5)), ones(1,floor(Int32, N/5))),vcat(rand(dist, 1,floor(Int32, N/5)), ones(1,floor(Int32, N/5)).*-1),
    vcat(ones(1,floor(Int32, N/5)), rand(dist, 1,floor(Int32, N/5))), vcat(ones(1,floor(Int32, N/5)).*-1, rand(dist, 1,floor(Int32, N/5))),
    vcat(zeros(1,floor(Int32, N/5)), rand(1,floor(Int32, N/5))))
    out_original = M(a)
    loss = (β/N)*sum(abs2, out_original)
    return loss
end

#add losses
function total_loss(N, space = 0.01, β = 750)
    return inner_loss(N, space) + bc_loss(floor(Int32, N/2), β)
end

#train model
function train_model(N = 100, train_time = 20)
    df = DataFrame(time = Int[], in_loss = Float64[],bound_loss = Float64[])
    c = 1
    n = floor(Int32, N/2)
    evalcb = function()
        @show(total_loss(N))
        in_loss = inner_loss(N, 0.01)
        bound_loss = bc_loss(n, 750)
        push!(df, [c, in_loss, bound_loss])
        c += 1
        if c == train_time
            CSV.write("$(c)_2d_cutsq.csv",df)
        end
    end
    opt = ADAM()
    dataset = [(N) for i=1:train_time]
    Flux.train!(total_loss, params(M), zip(dataset), opt, cb=throttle(evalcb, 10))
end

@epochs 1000 train_model(4000)

#plot the solution over a mesh in the domain
function plot_surface()
    data = zeros(201,201)
    grid_space = [i for i in -100:100]
    for i in grid_space
        for j in grid_space
            point = [i/100 j/100]
            @inbounds data[i+101,j+101] = M(point')[1]
        end
    end

    gr()
    heatmap(1:size(data,1),
        1:size(data,2), data,
        c = cgrad([:blue, :white,:red, :yellow]),
        xlabel = "X", ylabel = "Y",
        title = "2D Cut Square Fitted Solution")

    savefig("2d_cutsq_fitted.png")
end

plot_surface()
