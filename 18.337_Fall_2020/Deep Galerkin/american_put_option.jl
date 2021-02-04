using Flux
using Flux: @epochs
using Flux: params
using Flux: throttle
using Flux: glorot_uniform
using Flux: @functor
using Plots
using Distributions
using Zygote
using Flux.Optimise: update!
using ForwardDiff
using ReverseDiff
using FiniteDiff
using ZygoteRules

```
Workflow:

# 1. Define network architecture:
    # 1. define 1 LSTM-like layer
    # 2. define 1 dense layer with activation for the start
    # 3. define 1 dense layer without activation for the end
    # 4. connect them all together -> model
# 2. Define the finite element solution function
# 3. Define sampling function
# 4. Define the loss function
# 5. Train the model

```

# define a DGM layer
mutable struct LSTMLayer
    Ug :: Array{Float64}
    Uz :: Array{Float64}
    Ur :: Array{Float64}
    Uh :: Array{Float64}
    Wg :: Array{Float64}
    Wz :: Array{Float64}
    Wr :: Array{Float64}
    Wh :: Array{Float64}
    bg :: Array{Float64}
    bz :: Array{Float64}
    br :: Array{Float64}
    bh :: Array{Float64}
    act1
    act2
end

LSTMLayer(input_dim :: Integer, output_dim :: Integer, activation_1, activation_2) = LSTMLayer(glorot_uniform(output_dim, input_dim), glorot_uniform(output_dim, input_dim),
    glorot_uniform(output_dim, input_dim), glorot_uniform(output_dim, input_dim), glorot_uniform(output_dim, output_dim), glorot_uniform(output_dim, output_dim),
    glorot_uniform(output_dim, output_dim), glorot_uniform(output_dim, output_dim), zeros(output_dim), zeros(output_dim), zeros(output_dim), zeros(output_dim),
    activation_1, activation_2)

function (a :: LSTMLayer)(S,x)

    # assign all opbject variables
    Ug, Uz, Ur, Uh, Wg, Wz, Wr, Wh, bg, bz, br, bh, act1, act2 = a.Ug, a.Uz, a.Ur, a.Uh, a.Wg, a.Wz, a.Wr, a.Wh, a.bg, a.bz, a.br, a.bh, a.act1, a.act2

    # define all operations
    G = act1.(Ug*x + Wg*S .+ bg)
    Z = act1.(Uz*x + Wz*S .+ bz)
    R = act1.(Ur*x + Wr*S .+ br)
    H = act2.(Uh*x + Wh*(S.*R) .+ bh)
    S_new = (1 .- G).*H + Z.*S

    return S_new
end

# define dense layer with activation
mutable struct Dense_start
    W1 :: Array{Float64}
    b1 :: Array{Float64}
    act
end

Dense_start(input_dim::Integer, output_dim::Integer, act) = Dense_start(glorot_uniform(output_dim, input_dim), zeros(output_dim), act)

function (a :: Dense_start)(x)

    # assign all opbject variables
    W1, b1, act = a.W1, a.b1, a.act

    # define all operations
    S_1 = act.(W1*x .+ b1)

    return S_1
end

# define dense layer without activation
mutable struct Dense_end
    W :: Array{Float64}
    b :: Array{Float64}
end

Dense_end(input_dim::Integer, output_dim::Integer) = Dense_end(glorot_uniform(output_dim, input_dim), zeros(output_dim))

function (a :: Dense_end)(x)

    # assign all opbject variables
    W, b = a.W, a.b

    # define all operations
    y = W*x .+ b

    return y
end

# connect them together into one model
mutable struct DGM
    D1 :: Dense_start
    LSTMLayers :: Array{LSTMLayer}
    D2 :: Dense_end
end

DGM(layer_width, n_layers, spatial_dim) = DGM(Dense_start(spatial_dim + 1, layer_width, tanh),
[LSTMLayer(spatial_dim + 1, layer_width, tanh, tanh) for i in 1:n_layers], Dense_end(layer_width, 1))

function (a :: DGM)(t, x)

    # assign object variables
    D1, LSTMLayers, D2 = a.D1, a.LSTMLayers, a.D2

    #concatenate time and space inputs
    X = vcat(t, x)

    # define all operations
    S = D1(X)

    for lstm_layer in LSTMLayers
        S = lstm_layer(S, X)
    end

    y = D2(S)

    return y
end

# convey trainable parameters to Flux
@functor Dense_start
@functor LSTMLayer
@functor Dense_end
@functor DGM

# check if Flux recognizes parameters
m1 = DGM(50, 3, 1)
params(m1)

# define analytical solution function
function european_put(S, K, r, sigma, t)

   d1 = (log.(S./K) .+ (r + sigma^2/2)*(t_term-t))/(sigma*sqrt(t_term-t))
   d2 = d1 .- (sigma*sqrt(t_term-t))
   put_price = -S.*cdf.(Normal(0,1), -d1) .+ K*exp(-r * (t_term-t))*cdf.(Normal(0,1), -d2)

   return put_price
end

# define the fintie element solution for American put
function finite_element_solution()
    dt = t_term/n_steps
    t = collect(0:dt:t_term + dt)
    dx = sigma*sqrt(3*dt)
    alpha = 0.5*(sigma^2)*dt/(dx^2)
    beta = (r - 0.5*sigma^2)*dt/(2*dx)

    # diagnostics
    # println("r: $(r)")
    # println("sigma: $(sigma)")
    #
    # println("dt: $(dt)")
    # # println("t: $(t)")
    # println("dx: $(dx)")
    # println("alpha: $(alpha)")
    # println("beta: $(beta)")

    # log space grid
    x_max = ceil(10*sigma*sqrt(t_term)/dx)*dx
    x_min = -x_max
    x = collect(x_min:dx:x_max + dx)
    Ndx = length(x) - 1
    x_int = collect(2:Ndx)

    # diagnostics
    # println("x_max: $(x_max)")
    # println("x_min: $(x_min)")
    # # println("x: $(x)")
    # println("len(x): $(length(x))")
    # println("Ndx: $(Ndx)")
    # # println("x_int: $(x_int)")
    # println("len(x_int): $(length(x_int))")

    # put option price grid
    h = ones(Ndx + 1, n_steps + 1)
    h[:, end] = max.(K .- S0*exp.(x), 0)
    hc = ones(Ndx + 1)
    S = S0.*exp.(x)

    # # println("h: $(x)")
    # println("len(h): $(length(h))")
    # # println("hc: $(x)")
    # println("len(hc): $(length(hc))")
    # # println("S: $(x)")
    # println("len(S): $(length(S))")


    # calculate optimal exercise boundary
    exer_bd = ones(n_steps + 1)
    exer_bd[end] = K

    # back-step in time
    for n in collect(n_steps + 1:-1:2)
        # calculate continuation values
        hc[x_int] =  h[x_int, n] .- r*dt*h[x_int, n] .+ beta.*(h[x_int .+ 1, n] .- h[x_int .- 1, n]) + alpha*(h[x_int .+ 1, n] -2*h[x_int, n] + h[x_int .- 1,n])
        hc[1] = 2*hc[2] - hc[3]
        hc[Ndx + 1] = 2*hc[Ndx] - hc[Ndx]

        # compare with intrinsic values
        exer_val = K .- S
        h[:, n-1] = max.(hc, exer_val)

        # calculate optimal exercise boundaries
        checkIdx = (hc > exer_val)*1
        idx = argmax(checkIdx) - 1
        if max(checkIdx) > 0
            exer_bd[n-1] = S[idx]
        end
    end
    putPrice = h
    return t, S, putPrice, exer_bd
end

function sampling_function(space_internal, space_terminal)

    # within-domain sample
    t_internal = rand(Uniform(t_initial, t_term), space_internal)
    S_internal = rand(Uniform(S_low, S_high*S_multiplier), space_internal)

    # terminal sample
    t_terminal = t_term .*ones(space_terminal)
    S_terminal = rand(Uniform(S_low, S_high*S_multiplier), space_terminal)

    return t_internal', S_internal', t_terminal', S_terminal'
end


# define the loss function
function loss_function(N)

    t_internal, S_internal, t_terminal, S_terminal = sampling_function(nSim_interior, nSim_terminal)

    # differential operator loss
    ϵ = 0.01
    model_output_internal = model(t_internal, S_internal)
    ∂g∂x = (model(t_internal, S_internal .+ ϵ) - model_output_internal)./ϵ
    ∂g∂t = (model(t_internal .+ ϵ, S_internal) - model_output_internal)./ϵ
    ∂g∂xx = (model(t_internal, S_internal .+ 2*ϵ) - 2*model(t_internal, S_internal .+ ϵ) + model_output_internal)./(ϵ^2)

    operator_loss_vec = ∂g∂t + r.*S_internal.*∂g∂x + (0.5*(sigma^2)).*(S_internal.^2).*∂g∂xx - r.*model_output_internal

    payoff = relu.(K .- S_internal)
    value = model(t_internal, S_internal)
    L1 = mean((operator_loss_vec.*(value-payoff)).^2)

    temp = relu.(operator_loss_vec)
    L2 = mean(temp.^2)

    V_ineq = relu.(-(payoff - value))
    L3 = mean(V_ineq.^2)

    target_payoff = relu.(K .- S_terminal)
    fitted_payoff = model(t_terminal, S_terminal)

    L4 = mean((fitted_payoff - target_payoff).^2)

    return L1 + L2 + L3 + L4
end

# initialize all problem parameters
r = 0.05           # Interest rate
sigma = 0.5       # Volatility
K = 50             # Strike
t_term = 1              # Terminal time
S0 = 50           # Initial price

# Solution parameters (domain on which to solve PDE)
t_initial = 0 + 1e-10    # time lower bound
S_low = 0.0 + 1e-10  # spot price lower bound
S_high = 2*K         # spot price upper bound

# neural network parameters
n_steps = 10000
num_layers = 3
nodes_per_layer = 50
learning_rate = 0.001

# Training parameters
sampling_stages  = 100   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling

# Sampling parameters
nSim_interior = 1000
nSim_terminal = 100
S_multiplier  = 1.5   # multiplier for oversampling i.e. draw S from [S_low, S_high * S_multiplier]

# define model
model = DGM(nodes_per_layer, num_layers, 1)

S_model = reshape(S_values, 1, length(S_values))
t_model = 0.6667.*ones(1, length(S_values))

function train_model()
    # set optimizer as ADAM
    evalcb = function ()
    @show(loss_function())
    end
    opt = ADAM(learning_rate)
    dataset = [(1) for i in 1:10]
    Flux.train!(loss_function, params(model), zip(dataset), opt, cb = throttle(evalcb, 25))
end

@epochs 1000 train_model()

params(model)

# compare with analytical solution
compare_times = [0, 0.3333, 0.6667, 1]
n_plots = 100
S_values = [S_low + i*(S_high - S_low)/n_plots for i in 0:n_plots]
S_model = reshape(S_values, 1, length(S_values))

t, S, price, exer_bd = finite_element_solution()
S_idx = [i for i in 1:length(S) if(S[i] < S_high)]
t_idx = [1, 3334, 6668, n_steps]

for (i, t) in enumerate(compare_times)
    # analytical_soln(S, K, r, sigma, t)
    true_values = price[S_idx, t_idx[i]]
    t_model = t.*ones(1, length(S_values))
    fitted_values = model(t_model, S_model)
    fitted_values = reshape(fitted_values, length(fitted_values))
    display(plot!(S_values, fitted_values, label = "Fitted values"))
    display(plot!(S[S_idx], true_values, label = "True values"))
end
