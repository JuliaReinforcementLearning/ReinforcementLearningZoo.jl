export load_policy

using Pkg

function load_policy(s::String, T=AbstractPolicy)
    if isfile(s)
        RLCore.load(s, T)
    elseif isdir(s)
        load_policy(joinpath(s, "policy.bson"), T)
    else
        dir = Pkg.@artifact_str s
        return load_policy(dir, T)
    end
end