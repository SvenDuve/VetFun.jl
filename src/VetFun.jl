module VetFun


using Images
using ImageMagick
using FileIO
using Distributions
using Random
using HDF5
using JLD
using IterTools
using Flux
using Flux.Data.MNIST
using Statistics
using Flux: onehotbatch, onecold, logitcrossentropy
using Base.Iterators: partition
using Printf, BSON
using Parameters: @with_kw
using CUDAapi
using Flux.Data: DataLoader
using Flux: crossentropy, gradient, params, normalise, binarycrossentropy, onehot
using Flux.Optimise: update!, Descent
using MLBase
using Zygote
using BSON: @save
using BSON: @load

#include("models.jl")
#include(model3way.jl)


export extractFileList, 
        fotoReshape,
        extractDataLabels,
        returnClassImbalance,
        augFunc,
        createRandPics,
        extractClassMember,
        prepareImages,
        getData, 
        data_path,
        processX,
        train_3Cat_NN,
        model_3Cat



# @load "NNRoentgen3Way.bson" model
# @load "NNRoentgen3Way.bson" weights

# Flux.loadparams!(model, weights)

include("gettingData.jl")


end