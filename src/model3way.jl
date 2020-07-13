# using Flux
# using Flux.Data: DataLoader
# using Flux: crossentropy, gradient, params, logitcrossentropy, normalise, binarycrossentropy, onehot
# using Flux.Optimise: update!, Descent
# using MLBase
# using Images
# using ImageMagick
# using FileIO
# using Distributions
# using Random
# using HDF5
# using JLD
# using Plots


# function getData()

#     fileLocation = "data/"
#     return load(fileLocation * "roentgen.jld")

# end
# #sort out class imbalance


# data = getData()


# X_Data = data["x"]
# Y_Data = data["y"]





# allPositivesInd = findall(x -> x == 1, Y_Data)
# allNegativesInd = findall(x -> x == 0, Y_Data)

# allPositivesX = X_Data[allPositivesInd]
# allNegativesX = X_Data[allNegativesInd]


# allNegativesX_gen = [clamp01nan!(Float64.(imrotate(Gray.(allNegativesX[i] .+ randn()/5), randn()/3, axes(allNegativesX[i])))) .+ rand()/10 for i in 1:length(allNegativesX)];
# allNegativesX = vcat(allNegativesX, allNegativesX_gen)
# furtherNeg = rand(allNegativesX, 824)
# allNegativesX_gen = [clamp01nan!(Float64.(imrotate(Gray.(furtherNeg[i] .+ randn()/5), randn()/3, axes(allNegativesX[i])))) .+ rand()/10 for i in 1:length(furtherNeg)];
# allNegativesX = vcat(allNegativesX, allNegativesX_gen)




# # Put the data toghether
# X_Data = vcat(allNegativesX, allPositivesX)
# Y_Data = Int.(vcat(zeros(length(allNegativesX)), ones(length(allPositivesX))))


# Y_Data = Flux.onehotbatch(vec(Y_Data), [0, 1, 2])


# for i in 1:length(X_Data)
    
#     X_Data[i] = reshape(X_Data[i], 1, :)
    
# end


# # create a matrix from an Array of Arrays
# X_Data = vcat(X_Data...)


# # we want one picture to have dimension m features times 1 so m*1 per picture,
# # then the total matrix x should be m*n where m is the length of features and n is the number of examples
# # so also y is a 1 * n matrix


# X_Data = transpose(X_Data)
# #Y_Data = transpose(Y_Data)




# perm = shuffle(1:size(X_Data)[2])


# X_train, Y_train = X_Data[:, perm[1:Int(round(size(X_Data)[2] * 0.9))]], Y_Data[:, perm[1:Int(round(size(X_Data)[2] * 0.9))]]
# X_test, Y_test = X_Data[: , perm[Int(round(size(X_Data)[2] * 0.9)) + 1:end]], Y_Data[: , perm[Int(round(size(X_Data)[2] * 0.9)) + 1:end]]


# # sum(Y_train[1,:]) / size(Y_train)[2]
# # sum(Y_test[1,:]) / size(Y_test)[2]




# # size(X_train)
# # size(X_test)
# # size(Y_train)
# # size(Y_test)



# data_xr = DataLoader(X_train, Y_train, batchsize=1) 



# # data_xr.batchsize
# # data_xr.data
# # data_xr.imax
# # data_xr.indices
# # data_xr.nobs
# # data_xr.partial
# # data_xr.shuffle





# #method from baumann, check difference
# #data_xr_batch = (Flux.batch(X_train), Flux.batch(Y_train))

# model = Chain(Dense(size(data_xr.data[1])[1], 32, relu), Dense(32, 3, σ), softmax)


# params(model)


# #check model works:
# model(data_xr.data[1])


# loss(x, y) = Flux.crossentropy(model(x), y) #mse(model(x), y) # loss of the first m=1 examples
# #loss(x, y) = Flux.mse(model(x), y)

# #check loss function works
# loss(data_xr.data[1], data_xr.data[2])


# opt = Descent(0.001)
# @time Flux.train!(loss, params(model), data_xr, opt)


# for _ in 1:100

#     @time Flux.train!(loss, params(model), data_xr, opt)
#     println(loss(data_xr.data[1], data_xr.data[2]))


# end



# @save "NNRoentgen3Way.bson" model

# weights = params(model)

# @save "NNRoentgen3Way.bson" weights







#mean(model() .== Y_test)


# gt = Int64.(Y_test)
# gt = classify(Y_test) 
# pred = classify(model(X_test)) 
# #pred = vec(Int64.(model(X_test)))

# correctrate(gt, pred)
# errorrate(gt, pred)

# confusmat(3, gt , pred) # not sure about this, we have two classes, but doesnt like it

