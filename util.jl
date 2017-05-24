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
