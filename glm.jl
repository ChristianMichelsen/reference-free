using MLJ, CategoricalArrays, PrettyPrinting
using DataFrames
using UrlDownload

#%%

LinearRegressor = @load LinearRegressor pkg = GLM
LinearBinaryClassifier = @load LinearBinaryClassifier pkg = GLM


baseurl = "https://raw.githubusercontent.com/tlienart/DataScienceTutorialsData.jl/master/data/glm/"

dfX = DataFrame(urldownload(baseurl * "X3.csv"))
dfYbinary = DataFrame(urldownload(baseurl * "Y3.csv"))
dfX1 = DataFrame(urldownload(baseurl * "X1.csv"))
dfY1 = DataFrame(urldownload(baseurl * "Y1.csv"));


first(dfX, 3)
first(dfY1, 3)

ms = models() do m
    AbstractVector{Count} <: m.target_scitype
end
foreach(m -> println(m.name), ms)
