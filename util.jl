function build_vocab(entries)
    # trn = filter(ei->ei["split"]=="train", entries)
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
function mfsc(arr, sr=16000.0)
    feacalc(
        xx; defaults=:rasta, sadtype=:none, normtype=:mvn, augtype=:ddelta,
        dcttype=-1, sr=sr)
end
