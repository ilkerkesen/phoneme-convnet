function build_vocab(entries)
    c = 1
    p2i, i2p = Dict(), Dict()
    for sample in train
        phns = sample["phonemes"]
        for (_,_,phn) in phns
            if !haskey(p2i, phn)
                p2i[phn] = c
                i2p[c] = phn
            end
        end
    end
    return p2i, i2p
end

# TODO: implement this function
function make_input(samples)
end

# TODO: implement this function
function make_output(samples)
end



# feature extractor
function mfsc(arr; sr=16000.0, preprocess=true)
    if preprocess
        nch = 1
        ns = div(length(arr), nch)
        arr = reshape(arr, nch, ns)' / (1<<15)
    end

    feacalc(
        arr; defaults=:rasta, sadtype=:none, normtype=:mvn, augtype=:ddelta,
        dcttype=-1, sr=sr)
end
