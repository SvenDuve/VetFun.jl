
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



function fotoReshape(file::String)

    im = imresize(load(file), 32, 32);
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



function prepareImages(folders)

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

    fileLocation = "/Users/svenduve/.julia/dev/VetFun/data/"
    save(fileLocation * "roentgen.jld", "x", x, "y", y)

    return x, y

end





function getData()

    fileLocation = "data/"
    return load(fileLocation * "roentgen.jld")

end

