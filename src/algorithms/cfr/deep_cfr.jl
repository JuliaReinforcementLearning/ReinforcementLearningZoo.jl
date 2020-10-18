export DeepCFR

using Statistics: mean

"""
    DeepCFR(;kwargs...)

Symbols used here follow the paper: [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164)

# Keyword arguments

- `K`, number of traverrsal.
- `t`, number of iteration.
- `Π`, the policy network.
- `V`, a dictionary of each player's advantage network.
- `MΠ`, a strategy memory.
- `MV`, a dictionary of each player's advantage memory.
- `reinitialize_freq=1`, the frequency of reinitializing the value networks.
"""
Base.@kwdef mutable struct DeepCFR{TP, TV, TMP, TMV, I, R} <: AbstractCFRPolicy
    Π::TP
    V::TV
    MΠ::TMP
    MV::TMV
    K::Int                  = 20
    t::Int                  = 1
    reinitialize_freq::Int  = 1
    batch_size_V::Int       = 32
    batch_size_Π::Int       = 32
    n_training_steps_V::Int = 1
    n_training_steps_Π::Int = 1
    rng::R                  = Random.GLOBAL_RNG
    initializer::I          = glorot_uniform(rng)
end

function RLBase.get_prob(π::DeepCFR, env::AbstractEnv)
    I = send_to_device(device(π.Π), get_state(env))
    m = send_to_device(device(π.Π), get_legal_actions_mask(env))
    σ = softmax(π.Π(I) .+ ifelse.(m, 0.f0, -Inf32))
    send_to_host(σ)
end

"Run one interation"
function RLBase.update!(π::DeepCFR, env::AbstractEnv)
    for p in get_players(env)
        if p != get_chance_player(env)
            for k in 1:π.K
                external_sampling!(π, copy(env), p)
            end
            update_advantage_networks(π, p)
        end
    end
    π.t += 1
end

"Update Π (policy network)"
function RLBase.update!(π::DeepCFR)
    Π = π.Π
    D = device(Π)
    MΠ = π.MΠ
    for _ in 1:π.n_training_steps_Π
        batch_inds = rand(π.rng, 1:length(MΠ), π.batch_size_Π)
        I = send_to_device(D, Flux.batch([MΠ[i].I for i in batch_inds]))
        σ = send_to_device(D, Flux.batch([MΠ[i].σ for i in batch_inds]))
        t = send_to_device(D, Flux.batch([MΠ[i].t for i in batch_inds]))
        m = send_to_device(D, Flux.batch([MΠ[i].m for i in batch_inds]))
        gs = gradient(Flux.params(Π)) do 
            logits = Π(I) .+ ifelse.(m, 0f0, -Inf32)
            mean(reshape(t, 1, :) .* ((σ .- softmax(logits)) .^ 2))
        end
        update!(Π, gs)
    end
end

"Update advantage network"
function update_advantage_networks(π, p)
    V = π.V[p]
    MV = π.MV[p]
    if π.t % π.reinitialize_freq == 0
        for x in Flux.params(V)
            # TODO: inplace
            x .= π.initializer(size(x)...)
        end
    end
    if length(MV) >= π.batch_size_V
        for _ in 1:π.n_training_steps_V
            batch_inds = rand(π.rng, 1:length(MV), π.batch_size_V)
            I = send_to_device(device(V), Flux.batch([MV[i].I for i in batch_inds]))
            r̃ = send_to_device(device(V), Flux.batch([MV[i].r̃ for i in batch_inds]))
            t = send_to_device(device(V), Flux.batch([MV[i].t for i in batch_inds]))
            m = send_to_device(device(V), Flux.batch([MV[i].m for i in batch_inds]))
            gs = gradient(Flux.params(V)) do 
                mean(reshape(t, 1, :) .* ((r̃ .- V(I) .* m) .^ 2))
            end
            update!(V, gs)
        end
    end
end

"CFR Traversal with External Sampling"
function external_sampling!(π::DeepCFR, env::AbstractEnv, p)
    if get_terminal(env)
        get_reward(env, p)
    elseif get_current_player(env) == get_chance_player(env)
        env(rand(π.rng, get_actions(env)))
        external_sampling!(π, env, p)
    elseif get_current_player(env) == p
        V = π.V[p]
        s = get_state(env)
        I = send_to_device(device(V), Flux.unsqueeze(s, ndims(s)+1))
        A = get_actions(env)
        m = get_legal_actions_mask(env)
        σ = masked_regret_matching(V(I) |> vec, m)
        v = zeros(length(σ))
        v̄ = 0.
        for i in 1:length(m)
            if m[i]
                v[i] = external_sampling!(π, child(env, A[i]), p)
                v̄ += σ[i] * v[i]
            end
        end
        push!(π.MV[p],(I=s, t = π.t, r̃= v .- v̄, m = m))
        v̄
    else
        V = π.V[get_current_player(env)]
        s = get_state(env)
        I = send_to_device(device(V), Flux.unsqueeze(s, ndims(s)+1))
        A = get_actions(env)
        m = get_legal_actions_mask(env)
        σ = masked_regret_matching(V(I) |> vec, m)
        push!(π.MΠ,(I=s, t = π.t, σ=σ, m = m))
        a = sample(π.rng, A, Weights(σ, 1.0))
        env(a)
        external_sampling!(π, env, p)
    end
end

"This is the specific regret matching method used in DeepCFR"
function masked_regret_matching(v, m)
    v⁺ = max.(v .* m, 0.)
    s = sum(v⁺)
    if s > 0
        v⁺ ./= s
    else
        fill!(v⁺, 0.)
        v⁺[findmax(v, m)[2]] = 1.
    end
    v⁺
end
