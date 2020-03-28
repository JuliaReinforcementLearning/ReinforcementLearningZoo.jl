using Zygote
using Flux
using Flux.Optimise: update!

model = Chain(Dense(4,5),Dense(5,3),Dense(3,1),x->sigmoid.(x))  # Randomly initialized actor network
model5 = Chain(Dense(4,5),Dense(5,3),Dense(3,1),x->sigmoid.(x)) # Target actor network (Here I have not incorporated the exploration noise but can be developed later on)(Initializion)
model1 = Chain(Dense(4,3),Dense(3,2),Dense(2,1),x->sigmoid.(x))
model2 = Chain(Dense(1,3),Dense(3,2),Dense(2,1),x->sigmoid.(x))
model7 = Chain(Dense(4,3),Dense(3,2),Dense(2,1),x->sigmoid.(x))
model8 = Chain(Dense(1,3),Dense(3,2),Dense(2,1),x->sigmoid.(x))
model3 = Chain(x->(model1(x) .+ model2(model(x))),x->sigmoid.(x))  #Randomly initialized Critic Network
model6 = Chain(x->(model7(x) .+ model8(model5(x))),x->sigmoid.(x)) #Target Critic network(Initialization)
function Actor_Critic(tau , gamma, x , x1, r::Array{Float64,1},model,model1,model2,model3,model5,model6,model7,model8)    # Actor Critic Function where x is (i+1) th state and x1 is the i th state (Here I have implemented the network for learning a game in which input state is a $ element vector and action(or policy) is either 1 or 0 (Like a Cartpole game) . The whole model can be shifted to batch process later on when we will add replay buffer (R) to train original game . 
    opt=ADAM()
    y = Array{Float64,1}
    y = r .+ gamma.* model6(x)
    s = convert(Array{Float32,1},y) 
    function loss(x,y)               #loss function for updating critic network
         return sum((model3(x) .- y).^2)
    end
    grads = gradient(() -> loss(x1,s), Flux.params(model2,model1))
    update!(opt, Flux.params(model2,model1) , grads)     #updation of Critic Network
    for p = 1:3
       model7[p].b .= tau.*model1[p].b .+ (1 .- tau).*model7[p].b
       model7[p].W .= tau.*model1[p].W .+ (1 .- tau).*model7[p].W      # updation of target Critic Network
    end
    for p = 1:3
       model8[p].b .= tau.*model2[p].b .+ (1 .- tau).*model8[p].b
       model8[p].W .= tau.*model2[p].W .+ (1 .- tau).*model8[p].W
    end
    function loss1(x)                #loss function for updating actor network
        t = convert(Float32,model3(x)[1])
        return t
    end
    grads = gradient(() -> loss1(x1), Flux.params(model))
    update!(opt, Flux.params(model) , grads)           # Updation of Actor Network
    for p = 1:3
       model5[p].b .= tau.*model[p].b .+ (1 .- tau).*model5[p].b    # updation of target Actor Network
       model5[p].W .= tau.*model[p].W .+ (1 .- tau).*model5[p].W
    end 
    return model5(x1) , model6(x1) ,model,model1,model2,model3,model5,model6,model7,model8   
end
k = []
push!(k,1.0)
l = convert(Array{Float64,1},k)
Actor_Critic(0.01,0.1,rand(Float64,4),rand(Float64,4),l,model,model1,model2,model3,model5,model6,model7,model8)      #checking with random input

