using Flux,Zygote
using OpenAIGym
import Reinforce.action
import Flux.params
using LinearAlgebra
using Flux.Optimise: update!
using Distributions:Uniform
using Zygote:@adjoint
using Plots
plotly()

mutable struct PendulumPolicy <: Reinforce.AbstractPolicy
   train::Bool
   function PendulumPolicy(train=true)
     new(train)
   end
end

env = GymEnv("Pendulum-v0")

state_size = length(env.state)
action_size = length(env.actions)
action_bound = Float32(env.actions.hi[1])
batch_size = 25
mem_size = 1000000
gamma = 0.99
rho = 0.005
max_ep = 200
max_ep_len = 1000
max_frames = 12000
memory = []
frames = 0
eps = randn(action_size,batch_size)          #random noise

println(action_size)
println(state_size)
w_i(dims...) = rand(Uniform(-0.003,0.003),dims...)
actor_b = Chain(Dense(state_size, 20, relu),Dense(20,action_size,initW=w_i,initb=w_i))
actor_w = Chain(Dense(state_size, 20, relu),Dense(20,batch_size,initW=w_i,initb=w_i))   #actor network
reparam = Chain(Dense(state_size+action_size, 20, relu),Dense(20,20,relu),Dense(20,action_size,initW=w_i,initb=w_i))
actor = Chain(x->(actor_b(x) .+ eps*actor_w(x)),x->tanh.(x),x->x*action_bound) #reparametrisation network

critic1 = Chain(Dense(state_size+action_size, 100, relu),Dense(100,100,relu),Dense(100,1,initW=w_i,initb=w_i)) #Q function network 1
critic2 = Chain(Dense(state_size, 100, relu),Dense(100,100,relu),Dense(100,1,initW=w_i,initb=w_i))       #value fuction network
critic3 = Chain(Dense(state_size+action_size, 100, relu),Dense(100,100,relu),Dense(100,1,initW=w_i,initb=w_i))   #Q function network 2

#critic1_tgt = deepcopy(critic1)
critic2_tgt = deepcopy(critic2)    #target value network

function update_target!(target,model; rho = 0.005)
   for p = 1:3
       target[p].b .= rho*model[p].b .+ (1 - rho)*target[p].b    # updation of target value Network parameters
       target[p].W .= rho*model[p].W .+ (1 - rho)*target[p].W
   end 
end
function Log_likelihood(actor_w,actor_b,eps,alpha,x)
   llike1 = []
   d = reparam(vcat(x,eps))
   #g = actor_b(x) .+ d*actor_w(x)
   #k = zero(actor_w(x))
   k = sum((-log.(abs.(actor_w(x)))-(((d.-actor_b(x)).^2)./(2*((actor_w(x)).^2)))),dims=1) #log likelihood estimation 
   for i=1:batch_size
     h = k[i]
     for j=1:action_size
       h += log(1+(tanh(d[j,i]))^2)
     end
     push!(llike1,-h)
   end
   llike1 .= alpha*llike1
   llike = reshape(llike1,1,batch_size)
   return llike
end

@adjoint Log_likelihood(actor_w,actor_b,eps,alpha,x)=Log_likelihood_grad(actor_w,actor_b,eps,alpha,x), Δ -> Log_likelihood_grad(actor_w,actor_b,eps,alpha,x)' * Δ

function train()
   minibatch = sample(memory,batch_size)
   x = hcat(minibatch...)
   s = hcat(x[1,:]...)
   a = hcat(x[2,:]...)
   r = hcat(x[3,:]...)
   s1 = hcat(x[4,:]...)
   d = .!hcat(x[5,:]...)
   
   #a1 = actor(s1)
   #x1 = vcat(s1,a1)
   #q1 = critic1_tgt(x1)
   q2 = critic2_tgt(s1)
   y1 = r + gamma*q2.*d   
   x2 = vcat(s,a)
   v1 = critic1(x2)
   v3 = critic2(s)
   v2 = critic3(x2)
   y2 = ((sum(v1)<sum(v2) ? v1 : v2)-Log_likelihood(actor_w,actor_b,eps,0.005,s))
   loss_critic1 = Flux.mse(v1,y1)  
   loss_critic2 = Flux.mse(v3,y2)
   loss_critic3 = Flux.mse(v2,y1)
   opt = ADAM(0.001)
   grads1 = gradient(()->loss_critic1,Flux.params(critic1))
   grads2 = gradient(()->loss_critic2,Flux.params(critic2))
   grads3 = gradient(()->loss_critic3,Flux.params(critic3))
   update!(opt, Flux.params(critic1) , grads1)      #updation of Q network 1 params
   update!(opt, Flux.params(critic2) , grads2)      #updation of value network params
   update!(opt, Flux.params(critic3) , grads3)      #updation of Q network 2 params
 
   #model = Chain(x->actor(x),x->vcat(s,x),x->critic1(x))
   #a2 = actor(s)
   #x3 = vcat(s,a2)
   #v3 = critic1(x3)
   #v4 = critic3(x3)

   loss_policy = (sum(critic1(vcat(s,actor(s)))-Log_likelihood(actor_w,actor_b,eps,0.005,s)))/batch_size
   opt1 = ADAM(-0.001)
   grads3 = gradient(()->loss_policy,Flux.params(actor,actor_w,actor_b,reparam))
   update!(opt1, Flux.params(actor,actor_w,actor_b,reparam) , grads3)      #updating policy network and reparametrisation network params
end
function replay_buffer(state, action, reward, next_state, done)
    if length(memory)>= mem_size
        deleteat!(memory,1)
    end
    push!(memory,[state, action, reward, next_state, done])
end
function action(pie::PendulumPolicy, reward, state, action)        #action selector
    state = reshape(state, size(state)...,1)
    b = reparam(vcat(state,eps[:,1]))
    b1 = repeat(b,outer=(1,batch_size))
    actor = Chain(x->(actor_b(x) .+ b1*actor_w(x)),x->tanh.(x),x->x*action_bound)
    act_pred = actor(state)
    clamp.(act_pred[:,1],-action_bound,action_bound)
end
function episode!(env, pie=RandomPolicy())
    global frames
    ep = Episode(env, pie)
    frm = 0
    for (s,a,r,s1) in ep
      #OpenAIGym.render(env)
      r = env.done ? -1 : r
      if pie.train 
        replay_buffer(s,a,r,s1,env.done)
      end
      frames+=1
      frm+=1

      if length(memory) >= batch_size && pie.train
         train()
         #update_target!(critic1_tgt, critic1;rho=rho)
         update_target!(critic2_tgt, critic2;rho=rho)     #updating target value network params
      end
    end
    #print(ep.niter)
    ep.total_reward
end

x = []
y = []
sc = zeros(1000)
#training
for e1=1:max_ep
   idx = 1
   reset!(env)
   total_reward = episode!(env, PendulumPolicy())
   sc[idx] = total_reward
   idx = idx%1000 +1
   avg = mean(sc)
   #push!(x,e1)
   #push!(y,avg)
   println("Episode: $e1| Score: $total_reward|Avg score: $avg| Frames: $frames")
end
#display(plot(x,y))
#testing
for e2=1:200
   reset!(env)
   total_reward = episode!(env, PendulumPolicy(false))
   push!(x,e2)
   push!(y,total_reward)
   println("Episode: $e2| Score: $total_reward")
end
#display(plot(x,y))

