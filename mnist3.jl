## Classification of MNIST dataset
## with the convolutional neural network known as LeNet5.
## This script also combines various
## packages from the Julia ecosystem with Flux.
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Statistics, Random
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using ProgressMeter: @showprogress
import MLDatasets
import BSON

##

η = 3e-4             # learning rate
λ = 0                # L2 regularizer param, implemented as weight decay
batchsize = 128      # batch size
epochs = 20          # number of epochs
seed = 0             # set seed > 0 for reproducibility
use_cuda = true      # if true use cuda (if available)
infotime = 1         # report every `infotime` epochs
checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
tblogger_bool = true # log training with tensorboard
savepath = "runs/"   # results path


## utility functions
num_params(model) = sum(length, Flux.params(model))
round4(x) = round(x, digits = 4)


# LeNet5 "constructor".
# The model can be adapted to any image size
# and any number of output classes.
function LeNet5(; imgsize = (28, 28, 1), nclasses = 10)
    out_conv_size = (imgsize[1] ÷ 4 - 3, imgsize[2] ÷ 4 - 3, 16)

    return Chain(
        Conv((5, 5), imgsize[end] => 6, relu),
        MaxPool((2, 2)),
        Conv((5, 5), 6 => 16, relu),
        MaxPool((2, 2)),
        flatten,
        Dense(prod(out_conv_size), 120, relu),
        Dense(120, 84, relu),
        Dense(84, nclasses),
    )
end


function get_data()
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
    test_loader = DataLoader((xtest, ytest), batchsize = batchsize)

    return train_loader, test_loader
end

function eval_loss_accuracy(loader, model)
    l = 0.0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l / ntot |> round4, acc = acc / ntot * 100 |> round4)
end


## LOGGING UTILITIES
if tblogger_bool
    tblogger = TBLogger(savepath, tb_overwrite)
    set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
    @info "TensorBoard logging at \"$(savepath)\""
end


function report(epoch)
    train = eval_loss_accuracy(train_loader, model)
    test = eval_loss_accuracy(test_loader, model)
    println("Epoch: $epoch   Train: $(train)   Test: $(test)")
    if tblogger_bool
        set_step!(tblogger, epoch)
        with_logger(tblogger) do
            @info "train" loss = train.loss acc = train.acc
            @info "test" loss = test.loss acc = test.acc
        end
    end
end



Random.seed!(seed)


## DATA
train_loader, test_loader = get_data();
@info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

## MODEL AND OPTIMIZER
model = LeNet5();
@info "LeNet5 model: $(num_params(model)) trainable params"

ps = Flux.params(model);

opt = ADAM(η)
if λ > 0 # add weight decay, equivalent to L2 regularization
    opt = Optimiser(WeightDecay(λ), opt)
end


# loss
loss(ŷ, y) = logitcrossentropy(ŷ, y)


## TRAINING
@info "Start Training"
report(0)
for epoch = 1:epochs
    @showprogress for (x, y) in train_loader
        gs = gradient(ps) do
            ŷ = model(x)
            loss(ŷ, y)
        end

        Flux.Optimise.update!(opt, ps, gs)
    end

    ## Printing and logging
    epoch % infotime == 0 && report(epoch)
    if checktime > 0 && epoch % checktime == 0
        !ispath(savepath) && mkpath(savepath)
        modelpath = joinpath(savepath, "model.bson")
        BSON.@save modelpath model epoch
        @info "Model saved in \"$(modelpath)\""
    end
end