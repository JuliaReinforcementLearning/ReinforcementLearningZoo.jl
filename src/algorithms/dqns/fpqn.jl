export FPNLearner, FullyParametrizedNet

using Flux
using CUDA
using Random
using Zygote
using Statistics: mean
using LinearAlgebra: dot

"""
    ImplicitQuantileNet(;ψ, ϕ, header)

```
        quantiles (n_action, n_quantiles, batch_size)
           ↑
         header
           ↑
feature ↱  ⨀   ↰ transformed embedding
       ψ       ϕ
       ↑       ↑
       s        τ
```
"""
Base.@kwdef struct FullyParametrizedNet{A,B,C} <: AbstractApproximator
    ψ::A
    ϕ::B
    header::C
end

Flux.@functor FullyParametrizedNet


function (net::FullyParametrizedNet)(s, emb)
    #fpn = net.fpn_app(s)
    features = net.ψ(s)  # (n_feature, batch_size)
    emb_aligned = net.ϕ(emb)  # (n_feature, N * batch_size)
    merged =
        Flux.unsqueeze(features, 2) .*
        reshape(emb_aligned, size(features, 1), :, size(features, 2))  # (n_feature, N, batch_size)
    quantiles = net.header(flatten_batch(merged))
    reshape(quantiles, :, size(merged, 2), size(merged, 3))  # (n_action, N, batch_size)
end

"""
    FPNLearner(;kwargs)

See [paper](https://arxiv.org/abs/1806.06923)

# Keyworkd arugments
- `approximator`, a [`FullyParametrizedNet`](@ref)
- `target_approximator`, a [`FullyParametrizedNet`](@ref), must have the same structure as `approximator`
- `κ = 1.0f0`,
- `N = 32`,
- `N′ = 32`,
- `Nₑₘ = 64`,
- `K = 32`,
- `γ = 0.99f0`,
- `stack_size = 4`,
- `batch_size = 32`,
- `update_horizon = 1`,
- `min_replay_history = 20000`,
- `update_freq = 4`,
- `target_update_freq = 8000`,
- `update_step = 0`,
- `default_priority = 1.0f2`,
- `β_priority = 0.5f0`,
- `rng = Random.GLOBAL_RNG`,
- `device_seed = nothing`,
"""
Base.@kwdef mutable struct FPNLearner{A,T,B<:NeuralNetworkApproximator,R,D} <: AbstractLearner
    approximator::A
    target_approximator::T
    fpn_app::B
    sampler::NStepBatchSampler
    κ::Float32
    N::Int
    N′::Int
    Nₑₘ::Int
    K::Int
    γ::Float32
    stack_size::Union{Nothing,Int}
    batch_size::Int
    update_horizon::Int
    min_replay_history::Int
    update_freq::Int
    target_update_freq::Int
    update_step::Int
    default_priority::Float32
    β_priority::Float32
    rng::R
    device_rng::D
    loss::Float32
end

Flux.functor(x::FPNLearner) =
    (Z = x.approximator, Zₜ = x.target_approximator,f = x.fpn_app, device_rng = x.device_rng),
    y -> begin
        x = @set x.approximator = y.Z
        x = @set x.target_approximator = y.Zₜ
        x = @set x.fpn_app = y.f
        x = @set x.device_rng = y.device_rng
        x
    end

Flux.gpu(rng::MersenneTwister) = CUDA.CURAND.RNG()
Flux.cpu(rng::CUDA.CURAND.RNG) = MersenneTwister()

function FPNLearner(;
    approximator,
    target_approximator,
    fpn_app,
    κ = 1.0f0,
    N = 32,
    N′ = 32,
    Nₑₘ = 64,
    K = 32,
    γ = 0.99f0,
    stack_size = 4,
    batch_size = 32,
    update_horizon = 1,
    min_replay_history = 20000,
    update_freq = 4,
    target_update_freq = 8000,
    update_step = 0,
    default_priority = 1.0f2,
    β_priority = 0.5f0,
    rng = Random.GLOBAL_RNG,
    device_rng = CUDA.CURAND.RNG(),
    traces = SARTS,
    loss = 0.0f0,
)
    copyto!(approximator, target_approximator )  # force sync
    if device(approximator) !== device(device_rng)
        throw(ArgumentError("device of `approximator` doesn't match with the device of `device_rng`: $(device(approximator)) !== $(device_rng)"))
    end
    sampler = NStepBatchSampler{traces}(;γ=γ, n=update_horizon,stack_size=stack_size,batch_size=batch_size)
    FPNLearner(
        approximator,
        target_approximator,
        fpn_app,
        sampler,
        κ,
        N,
        N′,
        Nₑₘ,
        K,
        γ,
        stack_size,
        batch_size,
        update_horizon,
        min_replay_history,
        update_freq,
        target_update_freq,
        update_step,
        default_priority,
        β_priority,
        rng,
        device_rng,
        loss,
    )
end

function (learner::FPNLearner)(env)
    state = send_to_device(device(learner.approximator), get_state(env))
    state = Flux.unsqueeze(state, ndims(state) + 1)
    tau_i = cumsum(softmax(learner.fpn_app(state); dims=1),dims=1)
    ze = zeros(Float32,1,size(tau_i,2))
    tau = cat(ze,tau_i,dims=1)
    N = learner.N
    tau_hat = ones(Float32,N,size(tau,2))
    for i=1:N
        tau_hat[i,:]=(tau[i,:].+tau[i+1,:])./2
    end
    τ = tau #rand(learner.device_rng, Float32, learner.K, 1)
    τₑₘ = embed(τ, learner.Nₑₘ)
    quantiles = learner.approximator(state, τₑₘ)
    vec(mean(quantiles; dims = 2)) |> send_to_host
end

embed(x, Nₑₘ) = cos.(Float32(π) .* (1:Nₑₘ) .* reshape(x, 1, :))

function RLBase.update!(learner::FPNLearner, t::AbstractTrajectory)
    length(t[:terminal]) < learner.min_replay_history && return

    learner.update_step += 1

    if learner.update_step % learner.target_update_freq == 0
        copyto!(learner.target_approximator, learner.approximator)
    end

    learner.update_step % learner.update_freq == 0 || return

    inds, batch = sample(learner.rng, t, learner.sampler)

    if t isa PrioritizedTrajectory
        priorities = update!(learner, batch)
        t[:priority][inds] .= priorities
    else
        update!(learner,batch)
    end
end

function RLBase.update!(learner::FPNLearner, batch::NamedTuple)
    Z = learner.approximator
    Zₜ = learner.target_approximator
    f = learner.fpn_app
    N = learner.N
    N′ = learner.N′
    Nₑₘ = learner.Nₑₘ
    κ = learner.κ
    β = learner.β_priority
    batch_size = learner.batch_size

    D = device(Z)
    s, r, t, s′ = map(
        x -> send_to_device(D, x),
        (batch.states, batch.rewards, batch.terminals, batch.next_states),
    )

    tau_i_p = cumsum(softmax(f(s′); dims=1),dims=1)
    ze_p = zeros(Float32,1,batch_size)
    tau_p = cat(ze_p,tau_i_p,dims=1)
    tau_hat_p = ones(Float32,N,size(tau_p,2))
    for i=1:N
        tau_hat_p[i,:]=(tau_p[i,:].+tau_p[i+1,:])./2
    end
    τ′ = tau_hat_p[1:N,:] #rand(learner.device_rng, Float32, N′, batch_size)  # TODO: support β distribution
    τₑₘ′ = embed(τ′, Nₑₘ)
    zₜ = Zₜ(s′, τₑₘ′)
    #z = reshape(zₜ,size(zₜ,1),N+1,batch_size)
    avg_zₜ = zeros(Float32,size(zₜ,1),1,batch_size)
    p = zeros(Float32,1,batch_size)
    #tau_pf = tau_p[1:N,:]
    #for j=1:size(zₜ,1)
    for k=1:size(avg_zₜ,1)
        for j=1:batch_size
            c = 0
            for i=1:N
                c = c .+ (tau_p[i+1,j]-tau_p[i,j]).*zₜ[k,i,j]
            end
            avg_zₜ[k,1,j] = c 
        end
    end
    aₜ = argmax(avg_zₜ, dims = 1)
    aₜ = aₜ .+ typeof(aₜ)(CartesianIndices((0, 0:N-1, 0)))
    qₜ = reshape(zₜ[aₜ], :, batch_size)
    target = reshape(r, 1, batch_size) .+ learner.γ * reshape(1 .- t, 1, batch_size) .* qₜ
         

        #mean(zₜ, dims = 2)

    if !isnothing(batch.next_legal_actions_mask)
        masked_value = fill(typemin(Float32), size(batch.next_legal_actions_mask))
        masked_value[batch.next_legal_actions_mask] .= 0
        avg_zₜ .+= send_to_device(D, masked_value)
    end
     # reshape to allow broadcast

    tau_i = cumsum(softmax(f(s), dims=1),dims=1)
    ze = zeros(Float32,1,batch_size)
    tau = cat(ze,tau_i,dims=1)
    tau_hat = ones(Float32,N,size(tau,2))
    for i=1:N
        tau_hat[i,:]=(tau[i,:].+tau[i+1,:])./2
    end
    τ = tau_hat[1:N,:]#rand(learner.device_rng, Float32, N, batch_size)
    τₑₘ = embed(τ, Nₑₘ)
    a = CartesianIndex.(repeat(batch.actions, inner = N), 1:(N*batch_size))

    tau_em = embed(tau[1:N,:],Nₑₘ)
    #q_t = Z(s, tau_em)
    #q_t = reshape(q_t_i,:,batch_size)
    #z = reshape(zₜ,size(zₜ,1),N+1,batch_size)

    is_use_PER = !isnothing(batch.priorities)  # is use Prioritized Experience Replay
    if is_use_PER
        updated_priorities = Vector{Float32}(undef, batch_size)
        weights = 1.0f0 ./ ((batch.priorities .+ 1f-10) .^ β)
        weights ./= maximum(weights)
        weights = send_to_device(D, weights)
    end 
    gs1 = Zygote.gradient(Flux.params(f)) do 
        z1 = flatten_batch(Z(s, tau_em))
        z2 = flatten_batch(Z(s, τₑₘ))
        dW1 = zeros(Float32,N-1,batch_size)
        for i=2:N
            dW1[i,:] = 2 .* reshape(z1[a],N,batch_size)[i,:] .- reshape(z2[a],N,batch_size)[i,:] .- reshape(z2[a],N,batch_size)[i-1,:]
        end
        loss = Zygote.dropgrad(mean(sum(dW1,dims=1),dims=2)[1,1]) .* tau_em
    end
    update!(f, gs1)

    gs = Zygote.gradient(Flux.params(Z)) do
        z = flatten_batch(Z(s, τₑₘ))
        q = z[a]

        TD_error = reshape(target, N, 1, batch_size) .- reshape(q, 1, N, batch_size)
        # can't apply huber_loss in RLCore directly here
        abs_error = abs.(TD_error)
        quadratic = min.(abs_error, κ)
        linear = abs_error .- quadratic
        huber_loss = 0.5f0 .* quadratic .* quadratic .+ κ .* linear

        # dropgrad
        raw_loss =
            abs.(reshape(τ, 1, N, batch_size) .- Zygote.dropgrad(TD_error .< 0)) .*
            huber_loss ./ κ
        loss_per_quantile = reshape(sum(raw_loss; dims = 1), N, batch_size)
        loss_per_element = mean(loss_per_quantile; dims = 1)  # use as priorities
        loss =
            is_use_PER ? dot(vec(weights), vec(loss_per_element)) * 1 // batch_size :
            mean(loss_per_element)
        ignore() do
            # @assert all(loss_per_element .>= 0)
            is_use_PER && (
                updated_priorities .=
                    send_to_host(vec((loss_per_element .+ 1f-10) .^ β))
            )
            learner.loss = loss
        end
        loss
    end

    update!(Z, gs)

    is_use_PER ? updated_priorities : nothing
end
