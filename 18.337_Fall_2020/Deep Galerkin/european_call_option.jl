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
# 2. Define the analytical solution function
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
function analytical_soln(S, K, r, sigma, t)

   d1 = (log.(S./K) .+ (r + sigma^2/2)*(t_term-t))/(sigma*sqrt(t_term-t))
   d2 = d1 .- (sigma*sqrt(t_term-t))
   call_price = S.*cdf.(Normal(0,1), d1) .- K*exp(-r * (t_term-t))*cdf.(Normal(0,1), d2)

   return call_price
end

# define sampling function
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
function loss_function()

    t_internal, S_internal, t_terminal, S_terminal = sampling_function(nSim_interior, nSim_terminal)

    # differential operator loss
    model_output_internal = model(t_internal, S_internal)
    ϵ = 0.01

    ∂g∂x = (model(t_internal, S_internal .+ ϵ) - model_output_internal)./ϵ
    ∂g∂t = (model(t_internal .+ ϵ, S_internal) - model_output_internal)./ϵ
    ∂g∂xx = (model(t_internal, S_internal .+ 2*ϵ) - 2*model(t_internal, S_internal .+ ϵ) + model_output_internal)./(ϵ^2)
    operator_loss_vec = ∂g∂t + r.*S_internal.*∂g∂x + (0.5*(sigma^2)).*(S_internal.^2).*∂g∂xx - r.*model_output_internal
    operator_loss = sum(abs2, operator_loss_vec)

    # terminal condition  loss
    target_output_terminal = relu.(S_terminal .- K)
    model_output_terminal = model(t_terminal, S_terminal)
    terminal_loss = sum(abs2, model_output_terminal - target_output_terminal)

    return operator_loss + terminal_loss
end

# initialize all problem parameters
r = 0.05           # Interest rate
sigma = 0.25       # Volatility
K = 50             # Strike
t_term = 1              # Terminal time
S0 = 0.5           # Initial price

# Solution parameters (domain on which to solve PDE)
t_initial = 0 + 1e-10    # time lower bound
S_low = 0.0 + 1e-10  # spot price lower bound
S_high = 2*K         # spot price upper bound

# neural network parameters
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
# get params
θ = params(model)

function train_model()
    # set optimizer as ADAM
    evalcb = () -> @show(loss_function())
    opt = ADAM(learning_rate)
    dataset = [(1) for i in 1:20]
    Flux.train!(loss_function, params(model), zip(dataset), opt, cb = throttle(evalcb, 10))
end

@epochs 1000 train_model()

params(model)

# compare with analytical solution
compare_times = [t_initial, t_term/3, 2*t_term/3, t_term-0.1]
n_plots = 41
S_values = [2.5*i for i in 0:40]
S_model = reshape(S_values, 1, length(S_values))

for t in compare_times
    true_values = analytical_soln(S_values, K, r, sigma, t)
    t_model = t.*ones(1, length(S_values))
    fitted_values = model(t_model, S_model)
    fitted_values = reshape(fitted_values, length(fitted_values))
    display(plot(true_values, label = "Truth", linestyle = :dash, lw = 3, title = "t = $(t)",
    xlabel = "Stock prices", ylabel = "Call option prices"))
    display(plot!(fitted_values, label = "DGM", lw = 2))
    savefig("$(t)_euro_call.png")
end
