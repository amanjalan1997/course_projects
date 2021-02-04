using Flux
using Flux: @epochs
using Flux: params
using Flux: throttle
using Plots
using Distributions
using DataFrames
using CSV

@inline phi(x) = @. x/(1 + exp(-x))
@inline f(x) = 2*π^2 .*sum(cos.(π*x[i,:]') for i=1:2, dims=1)

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
    a = rand(Uniform(0,1),2,N)
    loss = 0
    l1 = [1, 0]
    l2 = [0, 1]

    @inbounds loss += (sum(0.5*((M(a .+ space.*l1) .- M(a))./space).^2)/N)[1]
    @inbounds loss += (sum(0.5*((M(a .+ space.*l2) .- M(a))./space).^2)/N)[1]
    @inbounds loss += (sum(0.5*(π^2 * (M(a)).^2))/N)[1]
    @inbounds loss += (sum(-f(a) .*M(a))/N)[1]

    return loss
end

function bc_loss(β, N = 1000)

    d = (rand(1)/100)
    point = zeros(2,1000)
    p = rand(1,100)
    point = hcat(hcat(zeros(250), rand(250))', hcat(ones(250), rand(250))', hcat(rand(250), zeros(250))', hcat(rand(250), ones(250))')
    loss = 0
    for i=1:2
        if i == 1
            l1 = [1, 0]
        else
            l1 = [0, 1]
        end
        @inbounds loss += β*(sum(((1/2*M(point[:,500*(i-1)+1:500*i] .+ d.*l1) .- 1/2*M(point[:,500*(i-1)+1:500*i] .- d.*l1))./d).^2)/100)[1]
    end
    return loss
end

#add losses
function total_loss(N, space = 0.01, β = 50)
    return inner_loss(N, space) + bc_loss(β)
end

function loss_true(xx=1,yy=1,xx1=1,xx2=2)
  points=rand(2,2000)
  F_true(x)= sum(cos.(pi*x[i,:]') for i=1:2, dims=1)
  errors=sqrt(sum((M(points)-F_true(points)).^2)/2000)
  return errors
end

#train model
function train_model(N = 100, train_time = 20)
    df = DataFrame(time = Int[], in_loss = Float64[],bound_loss = Float64[],true_loss = Float64[])
    c = 1
    evalcb = function()
        @show(total_loss(N), loss_true())
        in_loss = inner_loss(N, 0.01)
        bound_loss = bc_loss(50)
        true_loss = loss_true()
        push!(df, [c, in_loss, bound_loss, true_loss])
        c += 1
        if c == train_time
            CSV.write("$(c)_2d_neumann.csv",df)
        end
    end
    opt = ADAM()
    dataset = [(N) for i=1:train_time]
    Flux.train!(total_loss, params(M), zip(dataset), opt, cb=throttle(evalcb, 50))
end

@epochs 1000 train_model(4000)

#plot the solution over a mesh in the domain
F_true(x)= sum(cos.(pi*x[i,:]') for i=1:2, dims=1)
function plot_surface()
    data = zeros(100,100)
    grid_space = [i for i in 1:100]
    for i in grid_space
        for j in grid_space
            point = [i/100 j/100]
            data[i,j] = M(point')[1]
        end
    end

    gr()
    heatmap(1:size(data,1),
        1:size(data,2), data',
        c = cgrad([:blue, :white,:red, :yellow]),
        xlabel = "X", ylabel = "Y",
        title = "2D Neumann Fitted Solution")

    savefig("2d_neumann_fitted.png")


    for i in grid_space
        for j in grid_space
            point = [i/100 j/100]
            data[i,j] = F_true(point')[1]
        end
    end

    gr()
    heatmap(1:size(data,1),
        1:size(data,2), data',
        c = cgrad([:blue, :white,:red, :yellow]),
        xlabel = "X", ylabel = "Y",
        title = "2D Neumann True Solution")

    savefig("2d_neumann_true.png")
end

plot_surface()
