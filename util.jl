# three class for each phoneme
function build_phoneme_vocab(entries)
    c = 1
    p2i, i2p = Dict(), Dict()
    for sample in entries
        phns = sample["phonemes"]
        for (_,_,phn) in phns
            phnb = phn*"-b" # begin
            if !haskey(p2i, phnb)
                p2i[phnb] = c
                i2p[c] = phnb
                c += 1
            end

            phnm = phn*"-m" # middle
            if !haskey(p2i, phnm)
                p2i[phnm] = c
                i2p[c] = phnm
                c += 1
            end

            phne = phn*"-e" # end
            if !haskey(p2i, phne)
                p2i[phne] = c
                i2p[c] = phne
                c += 1
            end
        end
    end
    return p2i, i2p
end

# separate vocabulary builder
function build_vocab(entries; kk="phonemes")
    c = 1
    w2i, i2w = Dict(), Dict()
    for sample in entries
        words = sample[kk]
        for (_,_,w) in words
            if !haskey(w2i, w)
                w2i[w] = c
                i2w[c] = w
                c += 1
            end
        end
    end
    return w2i, i2w
end

function make_input(samples)
    return cat(4, map(si->si[1], samples)...)
end

function make_output(samples, p2i)
    return map(s->p2i[s[2]], samples)
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

# read features arrays from hdf5 file
function make_data(
    filepath, entries, split="train";
    remove_sa1=true, nframes=15, sr=16000.0, triphones=true)
    samples = filter(ei->ei["split"]==split, entries)
    if remove_sa1
        samples = filter(si->si["sample"] != "sa1", samples)
    end

    features = load(filepath, "features")
    data = []
    i = 0
    for sample in samples
        feature = features[sample["longid"]]'
        # i+=1; info("i=$i")
        ss = []
        ll = div(nframes-1,2)
        isr(x) = Int(round(x/(sr/100)))
        # @show sample["phonemes"]
        pp = map(p->(max(1,isr(p[1])), min(size(feature,2),isr(p[2])), p[3]), sample["phonemes"])
        # @show pp
        for k = max(1+ll,pp[1][1]):size(feature,2)-ll
            while !(k >= pp[1][1] && k <= pp[1][2])
                # @show k, pp[1][1], pp[1][2]
                shift!(pp)
                length(pp) == 0 && break
            end

            length(pp) == 0 && break
            ff = feature[:,k-ll:k+ll]
            ff = reshape(ff, 40, 1, nframes*3, 1)
            push!(ss, (ff, pp[1][3]))
        end

        # for (ti,tf,phn) in sample["phonemes"]
        #     ll = div(nframes-1,2)
        #     ff = zeros(size(feature,1), size(feature,2)+2*ll+1)
        #     ff[:,1+ll:end-1-ll] = feature

        #     ti0 = Int(round(ti/(sr/100)))
        #     tf0 = Int(round(tf/(sr/100)))
        #     tm = div(tf0-ti0,2)+ti0
        #     ll = div((nframes-1),2)

        #     # @show ti0,tf0,tm,ll,size(ff)
        #     push!(ss, (reshape(ff[:,tm:tm+2ll],40,1,nframes*3,1), phn))
        # end
        push!(data, ss)
    end

    return data
end

function get_features(filepath,entries)
    filepath = abspath(filepath)
    features = h5open(filepath, "w") do f
        data = Dict()
        for entry in entries
            data[entry["longid"]] = read(f, entry["longid"])
        end
        data
    end
    return features
end

# transition probabilities, a frequentist approach
function get_probabilities(jsondata, p2i, i2p; sr=16000.0)
    posteriors = zeros(Float32, length(p2i), length(p2i))
    priors = zeros(Float32, length(p2i))

    entries = filter(
        j->j["split"]=="train" && j["sample"] != "sa1", jsondata)
    for entry in entries
        isr(x) = Int(round(x/(sr/100)))
        segments = entry["phonemes"]
        segments = map(s->(isr(s[1]),isr(s[2]),s[3]), segments)
        for k = 1:length(segments)-1
            curr = segments[k]
            next = segments[k+1]

            dif = curr[2]-curr[1]
            priors[p2i[curr[3]]] += dif+1
            posteriors[p2i[curr[3]],p2i[next[3]]] += 1
            posteriors[p2i[curr[3]],p2i[curr[3]]] += dif
        end
        curr = segments[end]
        dif = segments[end][2]-segments[end][1]
        posteriors[p2i[curr[3]],p2i[curr[3]]] += dif
        priors[p2i[segments[end][3]]] += dif+1
    end

    # scale
    priors = priors / sum(priors)
    posteriors = posteriors ./ sum(posteriors,2)

    return posteriors, priors
end
