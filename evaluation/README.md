# medical entity linking evalutation

## Availible datasets

Scientific language datasets:
- NCBI-diseas `./eval_scripts_ncbi_bc5cdr.sh 0` (where `0` is the device index)
- BC5CDR-disease `./eval_scripts_ncbi_bc5cdr.sh 0`
- BC5CDR-chemical `./eval_scripts_ncbi_bc5cdr.sh 0`
- (MedMentions unavailible due to licensing issues of UMLS.)

Social-media language datasets:
- COMETA (stratified general) `./eval_scripts_cometa.sh 0`
- COMETA (zeroshot general) `./eval_scripts_cometa.sh 0`
- AskAPatient `./eval_scripts_askapatient_10cv.sh 0`

Download datasets from [Dropbox](https://www.dropbox.com/s/s33fxxg23ev59ic/mel-test-data.tar.gz?dl=0) and put the unzipped `data` under `evaluation/`.
