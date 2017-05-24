import os
import argparse
import scipy.io as sio
from scipy.io import wavfile
import simplejson as json
import numpy as np
import h5py

def read_entries(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    lines = map(lambda line: line.replace('\n',''), lines)
    f.close()
    xs = map(lambda x: x.split(' '), lines)
    xs = map(lambda x: (int(x[0]), int(x[1]), x[2]), xs)
    return xs


def main(datadir, outdir, tmpdir):
    jsonfile = os.path.join(outdir, 'timit.json')
    matfile = os.path.join(outdir, 'timit.mat')
    entries = list()
    wavearrays = dict()

    sid = 1
    for split in ('train', 'test'):
        splitid = 1
        splitdir = os.path.join(datadir, split)
        for drx in os.listdir(splitdir):
            speakers = os.listdir(os.path.join(splitdir, drx))
            for speaker in speakers:
                all_content = os.listdir(os.path.join(splitdir, drx, speaker))
                wavfiles, phnfiles, wrdfiles, txtfiles = map(
                    lambda x: filter(lambda y: str.endswith(y,x), all_content),
                    ('wav','phn','wrd','txt'))
                wavfiles.sort()
                phnfiles.sort()
                wrdfiles.sort()
                txtfiles.sort()

                samples = map(lambda x: os.path.splitext(x)[0], wavfiles)
                for i in range(len(samples)):
                    sample = samples[i]

                    # some validation
                    ns =  []
                    wavfilei = wavfiles[i]
                    ns.append(os.path.splitext(wavfilei)[0])

                    phnfile = phnfiles[i]
                    ns.append(os.path.splitext(phnfile)[0])

                    wrdfile = wrdfiles[i]
                    ns.append(os.path.splitext(wrdfile)[0])

                    txtfile = txtfiles[i]
                    ns.append(os.path.splitext(txtfile)[0])

                    for name in ns:
                        if name != sample:
                            raise "Validation error in data processing"

                    # run sox, so we will be able to read TIMIT sample
                    source = os.path.join(splitdir, drx, speaker, wavfilei)
                    target = os.path.join(tmpdir, wavfilei)
                    command = "sox {} {}".format(source, target)
                    os.system(command)

                    # read wavfile
                    f = open(target, 'r')
                    srate, wavdata = wavfile.read(f)
                    f.close()
                    os.remove(target)

                    # read phnfile
                    filepath = os.path.join(splitdir, drx, speaker, phnfile)
                    phonemes = read_entries(filepath)

                    # read wrdfile
                    filepath = os.path.join(splitdir, drx, speaker, wrdfile)
                    words = read_entries(filepath)

                    # read txtfile
                    filepath = os.path.join(splitdir, drx, speaker, txtfile)
                    sentence = read_entries(filepath)[0]

                    # sample id
                    longid = "{}-{}-{}-{}".format(split, drx, speaker, sample)

                    entry = {
                        'sid': sid,
                        'drx': drx,
                        'speaker': speaker,
                        'splitid': splitid,
                        'sample': sample,
                        'longid': longid,
                        'split': split,
                        'srate': srate,
                        'sentence': sentence,
                        'words': words,
                        'phonemes': phonemes
                    }

                    entries.append(entry)
                    wavearrays[longid] = wavdata

                    if sid % 100 == 0:
                        print '{} instances processed so far...'.format(sid)

                    sid += 1
                    splitid += 1

    # saving stuff
    sio.savemat(matfile, wavearrays)
    jsondata = json.dumps(entries)
    f = open(jsonfile, 'w')
    f.write(jsondata)
    f.close()

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", required=True, type=str)
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--tmpdir", required=True, type=str)
    args = parser.parse_args()

    datadir = os.path.abspath(args.datadir)
    outdir = os.path.abspath(args.outdir)
    tmpdir = os.path.abspath(args.tmpdir)

    if not os.path.exists(datadir):
        raise "Data path does not exist."

    if not os.path.exists(outdir):
        raise "Output path does not exist."

    if not os.path.exists(tmpdir):
        raise "Temperorary path does not exist."

    main(datadir, outdir, tmpdir)
