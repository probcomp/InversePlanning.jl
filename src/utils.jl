"Unzips an array of tuples to a tuple of arrays."
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

"Unzips dictionaries or arrays of pairs."
unzip_pairs(ps::AbstractDict) = unzip_pairs(collect(ps))
unzip_pairs(ps::AbstractArray{<:Pair}) = first.(ps), last.(ps)

"Normalize log weights."
lognorm(w) = w .- logsumexp(w)

"Convert vector of unnormalized scores to probabiities."
function softmax(scores)
    if isempty(scores) return Float64[] end
    ws = exp.(scores .- maximum(scores))
    z = sum(ws)
    return isnan(z) ? ones(length(scores)) ./ length(scores) : ws ./ z
end
