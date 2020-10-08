using Random

function RLCore.Experiment(::Val{:JuliaRL}, ::Val{:Minimax}, ::Val{:OpenSpiel}, game;)
    env = OpenSpielEnv(string(game))
    agents = (
        Agent(policy = MinimaxPolicy(), role = 0),
        Agent(policy = MinimaxPolicy(), role = 1),
    )
    hooks = (TotalRewardPerEpisode(), TotalRewardPerEpisode())
    description = """
      # Play `$game` in OpenSpiel with Minimax
      """
    Experiment(agents, env, StopAfterEpisode(1), hooks, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:TabularCFR},
    ::Val{:OpenSpiel},
    game;
    n_iter = 300,
    seed = 123,
)
    env = OpenSpielEnv(
        game;
        default_state_style = RLBase.Information{String}(),
        is_chance_agent_required = true,
    )
    rng = MersenneTwister(seed)
    π = TabularCFRPolicy(;rng = rng)

    description = """
      # Play `$game` in OpenSpiel with TabularCFRPolicy
      """
    Experiment(π, env, StopAfterStep(300), EmptyHook(), description)
end
