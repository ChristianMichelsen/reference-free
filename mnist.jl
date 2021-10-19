using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy
using Flux: @epochs
using Statistics
using MLDatasets

# Load the data
x_train, y_train = MLDatasets.MNIST.traindata()
x_valid, y_valid = MLDatasets.MNIST.testdata()

# Add the channel layer
x_train = Flux.unsqueeze(x_train, 3)
x_valid = Flux.unsqueeze(x_valid, 3)

# Encode labels
y_train = onehotbatch(y_train, 0:9)
y_valid = onehotbatch(y_valid, 0:9)

# Create the full dataset
train_data = DataLoader(x_train, y_train, batchsize = 128)


model = Chain(
    # 28x28 => 14x14
    Conv((5, 5), 1 => 8, pad = 2, stride = 2, relu),
    # 14x14 => 7x7
    Conv((3, 3), 8 => 16, pad = 1, stride = 2, relu),
    # 7x7 => 4x4
    Conv((3, 3), 16 => 32, pad = 1, stride = 2, relu),
    # 4x4 => 2x2
    Conv((3, 3), 32 => 32, pad = 1, stride = 2, relu),

    # Average pooling on each width x height feature map
    GlobalMeanPool(),
    flatten,
    Dense(32, 10),
    softmax,
)


# Getting predictions
ŷ = model(x_train)
# Decoding predictions
ŷ = onecold(ŷ)
println("Prediction of first image: $(ŷ[1])")




accuracy(ŷ, y) = mean(onecold(ŷ) .== onecold(y))
loss(x, y) = Flux.crossentropy(model(x), y)
# learning rate
lr = 0.1
opt = Descent(lr)
ps = Flux.params(model)


number_epochs = 10

for batch in train_data

    gradient = Flux.gradient(ps) do
        # Remember that inside the loss() is the model
        # `...` syntax is for unpacking data
        training_loss = loss(batch...)
        return training_loss
    end

    Flux.update!(opt, ps, gradient)
end

# @epochs number_epochs Flux.train!(loss, ps, train_data, opt)
accuracy(model(x_train), y_train)