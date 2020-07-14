# VetFun

[![Build Status](https://travis-ci.com/SvenDuve/VetFun.jl.svg?branch=master)](https://travis-ci.com/SvenDuve/VetFun.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/SvenDuve/VetFun.jl?svg=true)](https://ci.appveyor.com/project/SvenDuve/VetFun-jl)
[![Coverage](https://codecov.io/gh/SvenDuve/VetFun.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SvenDuve/VetFun.jl)
[![Coverage](https://coveralls.io/repos/github/SvenDuve/VetFun.jl/badge.svg?branch=master)](https://coveralls.io/github/SvenDuve/VetFun.jl?branch=master)


## Some Usage

Whole package heavily customized for X-Rays. 

1. ```model_3Cat(Args)``` runs a predefined model to predict lungs to suffer from either none, bacterial or viral infections. Takes ```Args``` for which learning rate, batchsize and epochs have to be determined. For example ```Args(0.001, 128, 100)```. Returns a Flux model.
2. ```saveModel(model, modelName)``` saves a model in .bson format in a relative path within the package. ```loadModel(modelName)``` loads the model into the current julia session. Have to run ```using Flux``` before.
3. ```makePrediction(img, modelName)``` allows predictions on prior unseen images.  