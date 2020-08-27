function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:Minimax},
    ::Val{:OpenSpiel},
    game;
)
    env = OpenSpielEnv(string(game))
    agents = (
        Agent(policy=MinimaxPolicy(), role=0),
        Agent(policy=MinimaxPolicy(), role=1)
    )
    hooks = (TotalRewardPerEpisode(), TotalRewardPerEpisode())
    description="""
    # Play `$game` in OpenSpiel with Minimax
    """
    Experiment(agents, env, StopAfterEpisode(1), hooks, description)
end

function RLCore.Experiment(
    ::Val{:JuliaRL},
    ::Val{:CFRPolicy},
    ::Val{:OpenSpiel},
    game;
    n_iter=300
)
    env = OpenSpielEnv(game;default_state_style=RLBase.Information{String}(), is_chance_agent_required=true)
    π = CFRPolicy(;n_iter=n_iter, env)

    agents = map(get_players(env)) do p
        if p == get_chance_player(env)
            Agent(;policy=RandomPolicy(), role=p)
        else
            Agent(;policy=π,role=p)
        end
    end

    hooks = [p == get_chance_player(env) ? EmptyHook() : TotalRewardPerEpisode() for p in get_players(env)]
    description="""
    # Play `$game` in OpenSpiel with CFRPolicy
    """
    Experiment(agents, env, StopAfterEpisode(10_000), hooks, description)
end