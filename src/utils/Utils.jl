module Utils

export is_integer, is_alpha

# include("helper.jl")

"""
    is_integer(s)
Return Whether or not the string s is an integer.
"""
function is_integer(s)
    return isa(tryparse(Int, s), Int)
end

"""
    is_alpha(s::String)
    Faster version of isletter function. Only works for ASCI, not unicode.
Return True or False.
"""
function is_alpha(text)
    all(c -> 'a' <= c <= 'z' || 'A' <= c <= 'Z', text)
end


end
