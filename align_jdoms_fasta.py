#!/usr/bin/env python3

import subprocess
import Bio.SeqIO

if __name__=="__main__":

    # Define the input and output filenames
    input_file = './filtered_jdoms_for_training.fasta'
    output_file = './filtered_jdoms_for_training_aligned.fasta'
    hmm_file = 'dnaj_hscb_djlc_aligned_filtered.hmm'

    # Read the input sequences from test.fasta
    input_seqs = list(Bio.SeqIO.parse(input_file, 'fasta'))

    # Perform HMM alignment for multiple sequences
    subprocess.run(['hmmalign','--outformat','A2M','--trim','-o','__tmp_ali.fasta', hmm_file, input_file], stdout=subprocess.DEVNULL)

    # Parse and process the aligned sequences
    seqs = Bio.SeqIO.parse('__tmp_ali.fasta', 'fasta')
    alignedSeqs = [''.join([x for x in s.seq if not x.islower()]).replace('.', '') for s in seqs]

    # Write the aligned sequences to the output file with original identifiers
    with open(output_file, 'w') as out_file:
        for i, seq in enumerate(alignedSeqs):
            print("aligning sequence",i+1)
            seq_record = Bio.SeqRecord.SeqRecord(
                Bio.Seq.Seq(seq), id=input_seqs[i].id, description=input_seqs[i].description)
            Bio.SeqIO.write(seq_record, out_file, 'fasta')

    print(f'Aligned sequences saved to {output_file}')
