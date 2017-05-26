function initweights(atype, hidden, ysize=61)
    w = []
    push!(w, xavier(8,1,45,150))
    push!(w, zeros(1,1,150,1))
    push!(w, xavier(6,1,150,300))
    push!(w, zeros(1,1,300,1))
    push!(w, xavier(1000,1500))
    push!(w, zeros(1000,1))
    push!(w, xavier(1000,1000))
    push!(w, zeros(1000,1))
    push!(w, xavier(ysize,1000))
    push!(w,  zeros(ysize,1))
    return map(wi->convert(atype,wi), w)
end

function predict(w,x; o=Dict())
    # conv1
    x = conv4(w[1],x; padding=0) .+ w[2]
    x = sigm(x)
    x = pool(x; window=(4,1), stride=(2,1))

    # conv2
    x = conv4(w[3],x; padding=0) .+ w[4]
    x = sigm(x)
    x = pool(x; window=(2,1), stride=(2,1))


    # mlp
    x = sigm(w[5] * mat(x) .+ w[6])
    x = dropout(x,get(o, :pdrop, 0.0))
    x = sigm(w[7] * x .+ w[8])
    x = dropout(x,get(o, :pdrop, 0.0))

    # softmax
    x = w[end-1] * x .+ w[end]
end

function train!(w,x,y,opts; o=Dict())
    values = []
    g = lossgradient(w,x,y; o=o, values=values)
    update!(w,g,opts)
    lossval = values[1]
    return lossval
end

function loss(w, x, ygold; o=Dict(), values=[])
    ypred = predict(w, x; o=o)
    lossval = -logprob(ygold,ypred)
    push!(values, lossval)
    return lossval / length(ygold)
end

lossgradient = grad(loss)

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    o1 = logp(ypred,1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

function softmax(y0)
    y1 = maximum(y0,1)
    y2 = y0 .- y1
    y3 = exp(y2)
    y4 = y3 ./ sum(y3,1)
end

function validate(w,val,p2i; o=Dict())
    batchsize = get(o, :batchsize, 100)
    lossval = losscnt = ncorrect = 0
    atype = get(o, :atype, gpu()>=0?KnetArray{Float32}:Array{Float32})
    for k = 1:batchsize:length(val)
        lower = k
        upper = min(length(val), k+batchsize-1)

        x = make_input(val[lower:upper])
        y = make_output(val[lower:upper], p2i)

        ypred = predict(w,convert(atype, x))
        lossval += -logprob(y,ypred)
        nrows,ncols = size(ypred)
        index = y + nrows*(0:(length(y)-1))
        ncorrect += sum(maximum(ypred,1).== ypred[index])
        losscnt += ncols
    end
    return lossval/losscnt, ncorrect/losscnt
end

function viterbi(ynorm, posteriors; stamp=5)
    ynorm = log(ynorm)
    posteriors = log(posteriors) # transitions
    timesteps = size(ynorm,2)
    x0 = ynorm[:,1]



    prev = []
    next = []
    for k = 1:length(x0)
        push!(prev, [x0[k], []])
        push!(next, [-Inf, []])
    end

    # log prob, path
    for t = 2:stamp:timesteps
        info("time=$t")
        for i = 1:size(ynorm,1) # onceki
            for j = 1:size(ynorm,1) # sonraki
                this_logprob = posteriors[i,j] + ynorm[j,t] + prev[i][1]
                if this_logprob > next[j][1]
                    next[j][1] = this_logprob
                    next[j][2] = [prev[i][2]..., j]
                end
            end
        end

        for k = 1:length(x0)
            prev[k][1] = next[k][1]
            prev[k][2] = copy(next[k][2])
            next[k][1] = -Inf
            empty!(next[k][2])
            gc()
        end
    end

    probs = map(ni->ni[1], next)
    el = indmax(probs)
    return next[el]
end
