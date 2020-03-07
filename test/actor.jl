using Flux
using Zygote

"""struct Actor
   state_size::Int8
   action_size::Int64
   layer_size::Array{Int32,1,3}
   learning_rate::Float64
   batch_size::Int32
   tau::Float64
    Connstructor for the Actor Network
     state_size:An integer denoting the dimensionality of the states
            in the current problem
     action_size: An integer denoting the dimensionality of the
            actions in the current problem
      hidden_units: An iterable defining the number of hidden units in
            each layer
      learning_rate: A fload denoting the speed at which the network
           will learn. default
      batch_size: An integer denoting the batch size 
      tau: A flot denoting the rate at which the target model will
            track the main model. Formally, the tracking function is defined as:
              target_weights = tau * main_weights + (1 - tau) * target_weights 
end"""
Actor1 = [5 6 4]
dict1 = Dict("W" => rand(Actor1[2],Actor1[1]),"b"=> rand(Actor1[2]))
dict2 = Dict("W" => rand(Actor1[3],Actor1[2]),"b"=> rand(Actor1[3]))
dict5 = Dict("W" => rand(Actor1[2],Actor1[1]),"b"=> rand(Actor1[2]))
dict6 = Dict("W" => rand(Actor1[3],Actor1[1]),"b"=> rand(Actor1[3]))
x = rand(5)
y = rand(4)
function generate_model(x, W1 ,W2,b1,b2)
      """ Generates the model based on the hyperparameters defined in the constructor"""
       function sigmoid(x)
           return 1/(1+exp(-x))
       end   
       W1 = dict1["W"]
       b1 = dict1["b"]
       layer1(x) = W1 * x .+ b1
       p = sigmoid.(layer1(x))

       W2 = dict2["W"]
       b2 = dict2["b"]
       layer2(x) = W2 * x .+ b2
       model(x) = layer2(p)
       
       """function loss(x,y)
             fhat(x) = model(x)
             yhat = fhat(x) # our prediction for y
             return sum((y-yhat).^2)
       end"""
       loss(x,y) = sum((y-model(x).^2))
       l5 = loss(x,y)
       grads = gradient(() -> loss(x, y), Params([dict1["W"],dict1["b"],dict2["W"],dict2["b"]]))
       dict3 = Dict("gW" => grads[dict1["W"]],"gb"=>grads[dict1["b"]] ) 
       dict4 = Dict("gW" => grads[dict2["W"]],"gb"=>grads[dict2["b"]] ) 
       println(loss(x,y))
       return model(x),dict3,dict4,l5
end
function train(x,dict1,dict2,dict5,dict6)
         """Updates the weights of the main network
            states: The states of the input to the network
            action_gradients: The gradients of the actions to update the
            network"""
              "model = Flux.Chain(Dense(6, 64, relu),Dense(64, 16, relu),Dense(16, 1, relu));
              x = rand(Bool, 14*16+4)
              y = 100"
              "opt = ADAM(params(model))
              Flux.@epochs 100 Flux.train!(loss, [(x,y)], opt)
         model1,model1_weight,model1_input = generate_model()"
            for i in 1:100
                 model,l1,l2,k1 = generate_model(x,dict1["W"],dict2["W"],dict1["b"],dict2["b"])
                 target_model,l3,l4,k2 = generate_model(x ,dict5["W"],dict6["W"],dict5["b"],dict6["b"]) 
            
                 dict1["W"] .-= 0.001 .* l1["gW"]
                 dict1["b"] .-= 0.001 .* l1["gb"]
                 dict2["W"] .-= 0.001 .* l2["gW"]
                 dict2["b"] .-= 0.001 .* l2["gb"]

                 dict5["W"] = 0.01 .* dict1["W"] .+ (1- 0.01) .* dict5["W"]
                 dict5["b"] = 0.01 .* dict1["b"] .+ (1- 0.01) .* dict5["b"]
                 dict6["W"] = 0.01 .* dict2["W"] .+ (1- 0.01) .* dict6["W"]
                 dict6["b"] = 0.01 .* dict2["b"] .+ (1- 0.01) .* dict6["b"]
                 
                 println(k1)
           end
end
train(x,dict1,dict2,dict5,dict6)
