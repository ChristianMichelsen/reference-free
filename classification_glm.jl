using DataFrames
using StatsBase
using BioSequences
using Serialization
using GLM

#%%

filename_out = "./df.data"
object = deserialize(filename_out)
df = object.df

# filename_out = "./df_ohe.data"
# object = deserialize(filename_out)
# df_ohe = object.df_ohe

max_columns = 1


#%%

value_counts(df, col) = combine(groupby(df, col), nrow)
# value_counts(df, :x1)
# value_counts(df, :x5)

levels = [DNA_A, DNA_C, DNA_G, DNA_T]
# levels = [DNA_C, DNA_A, DNA_G, DNA_T]
coding = DummyCoding(levels = levels)
# coding = StatsModels.FullDummyCoding()


variable_names = names(df, Not(:y))[1:max_columns]

contrast = Dict(Symbol(name) => coding for name in variable_names)


# formula = @formula(y ~ 1 + x1 * x2 * x3)
formula = term(:y) ~ sum(term.([1; variable_names]))


logistic = glm(formula, df, Bernoulli(), LogitLink(); contrasts = contrast)
# exp.(coef(logistic))

prediction = predict(logistic, df);
prediction_class = map(x -> Int(x > 0.5), prediction);


prediction_df = DataFrame(
    y_actual = df.y,
    y_predicted = prediction_class,
    prob_predicted = prediction,
    correctly_classified = df.y .== prediction_class,
);
# prediction_df.correctly_classified = prediction_df.y_actual .== prediction_df.y_predicted

accuracy = mean(prediction_df.correctly_classified)
println(
    "Using $(length(variable_names)) variables, the accuracy is ",
    round(100 * accuracy, digits = 2),
    "%",
)


#%%


# variable_names = names(df_ohe, Not(:y))
# formula = term(:y) ~ sum(term.([1; variable_names]))
# logistic = glm(formula, df_ohe, Bernoulli(), LogitLink())
# # exp.(coef(logistic))
