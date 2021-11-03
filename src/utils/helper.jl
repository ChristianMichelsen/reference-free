
# """
#     is_integer(s::String)
# Return Whether or not the string s is an integer.
# """
# function is_integer(s::String)
#     return isa(tryparse(Int, s), Int)
# end

# """
#     is_alpha(s::String)
#     Faster version of isletter function. Only works for ASCI, not unicode.
# Return True or False.
# """
# function is_alpha(text::String)
#     all(c -> 'a' <= c <= 'z' || 'A' <= c <= 'Z', text)
# end
