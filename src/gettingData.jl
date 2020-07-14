
function extractFileList(folders::Array{String,1})


    # create empty vector to hold the file names
    fileList = []

    # loop over the folders containing the files, and add the file path to the vector
    for folder in folders
        for file in readdir(folder)
            push!(fileList, folder * file)
        end
    end

    return fileList;


end



function fotoReshape(file::String; dims = (32, 32))

    im = imresize(load(file), dims...);
    Float32.(im)

end



function extractDataLabels(fileList)

    x = []
    y = []
    tagNormal::String = "NORMAL"
    tagPneumonia::String = "PNEUMONIA"

    for file in fileList
        try
            #resizeIm = imresize(load(file), 224, 224)
            push!(x, fotoReshape(file)); #Float64.(resizeIm)) #reshape(Float64.(resizeIm), 1, :))
            
        catch
            continue
        end



        if occursin(tagNormal, file)
            push!(y, 1);
        elseif occursin(tagPneumonia, file) && occursin("bacteria", file)
            push!(y, 2);
        elseif occursin(tagPneumonia, file) && occursin("virus", file)
            push!(y, 3);
        else
            error("This file is undetermined")
        end
        
    end

    return x, y


end




function returnClassImbalance(y)

    levels = sort(unique(y)) # return the different levels and sort

    if length(levels) != length(1)

        ratios = [] 
        ratios = [sum(y .== level) / length(y) for level in levels] # calculate the relative frequency of the levels wrt the total #


    else

        error("There is only one level!")

    end


    return ratios, levels


end


function augFunc(pic)
    
    newPic = clamp01nan!(imrotate(Gray.(pic .+ randn()/5), randn()/3, axes(pic))) .+ rand()/10
    return Float32.(newPic)

end


function createRandPics(classMembers, multiplicator)

    numPics = Int(round(multiplicator * length(classMembers)))
    pics = rand(classMembers, numPics)
    return [Float32.(augFunc(pic)) for pic in pics]

end




function extractClassMember(y, level)

    findall(x -> x == level, y)

end



function prepareImages(folders, fileName)

    add_xs = []
    add_ys = []

    fileList = extractFileList(folders)

    x, y = extractDataLabels(fileList)

    ratios, levels = returnClassImbalance(y)

    for i in 1:length(levels)

        idx = []
        idx = extractClassMember(y, levels[i])

        multiplicator = (maximum(ratios) / ratios[i]) - 1
        
        if multiplicator > 0

            x_rdm = createRandPics(x[idx], multiplicator)
            y_rdm = fill(levels[i], length(x_rdm))
            push!(add_xs, x_rdm)
            push!(add_ys, y_rdm)

        end

    end

    x = vcat(x, vcat(add_xs...))
    y = vcat(y, vcat(add_ys...))


    perm = shuffle(1:length(x))

    x = x[perm]
    y = y[perm]

    #fileLocation = "data/"
    save(joinpath(data_path(), fileName), "x", x, "y", y)
    #save(path_fileName, "x", x, "y", y)

    return x, y

end


data_path() = abspath(joinpath(@__DIR__, "..", "data"))

#, "roentgen.jld"))


function getData(fileLocation, fileName)

    return load(joinpath(fileLocation, fileName))

end



function processX(x)

    for i in 1:length(x)

        x[i] = reshape(x[i], 1, :)

    end


    x = vcat(x...)
    transpose(x)


end


function train_3Cat_NN(xs, ys, lr, batchsize, path)

    data = DataLoader(xs, ys, batchsize=batchsize)
    model = Chain(Dense(size(data.data[1])[1], 32, relu), Dense(32, 3, σ), softmax)
    loss(x, y) = Flux.crossentropy(model(x), y)
    opt = Descent(lr)


    for i in 1:5

        Flux.train!(loss, params(model), data, opt)
        println("Iteration: ", i, " loss ", loss(data.data[1], data.data[2]))
        
        
    end


    weights = params(model)

    @save path model

end


function model_3Cat()

    data = getData(data_path(), "roentgen.jld")
    #data = getData("/Users/svenduve/.julia/dev/VetFun/data/", "roentgen.jld")

    x_data = data["x"]
    y_data = data["y"]

    ratios, levels = returnClassImbalance(y_data)

    println("We have classes ", levels)
    println("We have ratios ", ratios)

    y_data = Flux.onehotbatch(vec(y_data), levels)
    x_data = processX(x_data)

    perm = shuffle(1:size(x_data)[2])

    xs, ys = x_data[:, perm[1:Int(round(size(x_data)[2] * 0.9))]], y_data[:, perm[1:Int(round(size(x_data)[2] * 0.9))]]
    x_test, y_test = x_data[: , perm[Int(round(size(x_data)[2] * 0.9)) + 1:end]], y_data[: , perm[Int(round(size(x_data)[2] * 0.9)) + 1:end]]

    train_3Cat_NN(xs, ys, 0.001, 1, joinpath(data_path(), "/NNRoentgen3Way.bson"))
    

    # @load "NNRoentgen3Way.bson" model
    # @load "NNRoentgen3Way.bson" weights
    @load joinpath(data_path(), "/NNRoentgen3Way.bson") model



    gt = classify(y_test) 
    pred = classify(model(x_test)) 
    println("There Correctrate is: ", correctrate(gt, pred))

    confusmat(3, gt , pred) # not sure about this, we have two classes, but doesnt like it

    

end