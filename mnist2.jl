using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using MLDatasets
#%%


#%%

# model = Chain(
#     Conv((5, 5), 1 => 8, pad = 2, stride = 2, relu), # 28x28 => 14x14
#     Conv((3, 3), 8 => 16, pad = 1, stride = 2, relu), # 14x14 => 7x7
#     Conv((3, 3), 16 => 32, pad = 1, stride = 2, relu), # 7x7 => 4x4
#     Conv((3, 3), 32 => 32, pad = 1, stride = 2, relu),
#     GlobalMeanPool(), # Average pooling on each width x height feature map

#     #x -> reshape(x, :, size(x, 4)), # old way
#     flatten,
#     Dense(32, 10),
#     softmax,
# );

#%%

η = 3e-4       # learning rate
batchsize = 256    # batch size
epochs = 10        # number of epochs

#%%

# Loading Dataset
xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtest, ytest = MLDatasets.MNIST.testdata(Float32)

# Reshape Data in order to flatten each image into a linear array
xtrain = Flux.flatten(xtrain)
xtest = Flux.flatten(xtest)

# One-hot-encode the labels
ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

# Create DataLoaders (mini-batch iterators)
train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
test_loader = DataLoader((xtest, ytest), batchsize = batchsize)

#%%

# Construct model
imgsize = imgsize = (28, 28, 1)
nclasses = 10
model = Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses))

ps = Flux.params(model) # model's trainable parameters

## Optimizer
opt = ADAM(η)


#%%

function loss_and_accuracy(data_loader, model)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        ŷ = model(x)
        ls += logitcrossentropy(ŷ, y, agg = sum)
        acc += sum(onecold(ŷ) .== onecold(y))
        num += size(x)[end]
    end
    return ls / num, acc / num
end


## Training
for epoch = 1:epochs
    for (x, y) in train_loader
        gs = gradient(() -> logitcrossentropy(model(x), y), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
    end

    # Report on train and test
    train_loss, train_acc = loss_and_accuracy(train_loader, model)
    test_loss, test_acc = loss_and_accuracy(test_loader, model)
    println("Epoch=$epoch")
    println("  train_loss = $train_loss, train_accuracy = $train_acc")
    println("  test_loss = $test_loss, test_accuracy = $test_acc")
end