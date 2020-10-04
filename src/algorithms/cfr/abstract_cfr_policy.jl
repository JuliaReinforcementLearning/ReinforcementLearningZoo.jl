abstract type AbstractCFRPolicy <: AbstractPolicy end

function Base.run(p::AbstractCFRPolicy, env, stop_condition, hook)
    @assert NumAgentStyle(env) isa MultiAgent
    @assert DynamicStyle(env) === SEQUENTIAL
    @assert RewardStyle(env) === TERMINAL_REWARD
    @assert ChanceStyle(env) === EXPLICIT_STOCHASTIC
    @assert DefaultStateStyle(env) isa Information

    while true
        update!(p, env)
        hook(POST_ACT_STAGE, p, env)
        stop_condition(p, env) && break
    end
end