# Biomedical Entity Linking Evalutation

## Monolingual

**Scientific language datasets:**
- NCBI-disease `./eval_scripts_ncbi_bc5cdr.sh 0` (where `0` is the device index)
- BC5CDR-disease `./eval_scripts_ncbi_bc5cdr.sh 0`
- BC5CDR-chemical `./eval_scripts_ncbi_bc5cdr.sh 0`
- MedMentions unavailible due to licensing issues of UMLS.
  -  To obtain MedMentions dictionary file, please download the UMLS2017AA version's `MRCONSO.RRF` file and extract the CUI-name pairs in the following format:
```
C0079564||htlv-ii rex protein
C0162999||oleoylamine
C0347197||benign mouth neoplasm
...
```
In the end, there should be 3,415,665 individual CUIs and 14.8M lines. Some of the lines are duplicated. After de-duplication, there should be around 7.4M individual lines which we used as the final reference dictionary.

**Social-media language datasets:**
- COMETA (stratified general) `./eval_scripts_cometa.sh 0`
- COMETA (zeroshot general) `./eval_scripts_cometa.sh 0`
- AskAPatient `./eval_scripts_askapatient_10cv.sh 0`

Download datasets from [here](https://www.dropbox.com/s/s33fxxg23ev59ic/mel-test-data.tar.gz?dl=0) and put the unzipped `data` under `evaluation/`.

## Cross-Lingual (XL-BEL)

To evaluate, run `./eval_scripts_xlbel_test.sh 0`. Change model path, language, etc. in the script.

`xl_bel/xlbel_v0.0` contains mentions and their CUIs, without contexts (for replicating numbers in [\[Liu et al., ACL 2021\]](http://fangyuliu.me/media/pdfs/xlbel_acl2021_preprint.pdf), this is all you need). 


For XL-BEL with context, please use `xl_bel/xlbel_v1.0`. Note that the target entity is marked with `<tgt>TARGET</tgt>`. 
