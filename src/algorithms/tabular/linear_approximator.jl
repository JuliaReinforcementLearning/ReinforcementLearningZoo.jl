export LinearVApproximator, LinearQApproximator

using LinearAlgebra: dot

struct LinearApproximator{N,O} <: AbstractApproximator
    weights::Array{Float64,N}
    optimizer::O
end

const LinearVApproximator = LinearApproximator{1}

LinearVApproximator(;n, init=0., opt=Descent(1.0)) = LinearApproximator(fill(init, n), opt)

(V::LinearVApproximator)(s) = dot(s, V.weights)

function RLBase.update!(V::LinearVApproximator, correction::Pair)
    w = V.weights
    s, Δ = correction
    w̄ = s .* Δ
    Flux.Optimise.update!(V.optimizer, w, w̄)
end