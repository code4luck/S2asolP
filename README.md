# S2asolP: Sequence-Structure aware Protein Solubility Prediction
code and data for S2asolP

## Installation
- Install Anaconda (https://www.anaconda.com/download)
    create S2asolP environment. (```conda env create -f s2asolp.yaml```)
- Install SCRATCH-1D<sup><a href="#ref1">[1]</a></sup> release 1.2 (http://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz)

- R requirements (https://www.r-project.org)
    - R libraries
        * bio3d
        * stringr
        * Interpol
        * zoo
- Download the sa-prot<sup><a href="#ref2">[2]</a></sup> model as encoder from https://huggingface.co/westlake-repl/SaProt_650M_PDB and place it into ```model``` folder.
You can create bio environment by conda. (```conda env create -f bio.yaml```)
Use ```conda activate s2asolp``` or ```conda activate R``` to activate the environment.

## Use S2asolP to predict test result
We have placed the computed results in the ```infer_res``` folder.
1. Download S2asolP data and model in https://drive.google.com/drive/folders/1SqC5NWzTx_McoL9l6KlUwWYmon8E-4mF?usp=sharing
2. Then unzip the downloaded data and place it into the ```data``` folder, and move ```s2asolp_checkpoint.pt``` into the ```checkpoints``` folder. 
3. Activate conda env (```source activate s2asolp```).
4. Run the bash_infer.sh (```Bash bash_infer.sh```).
## Use S2asolP to predict new test file
You need to perform the following steps to predict new test file (e.g. test_seq.fasta).
* Run SCRATCH with the new test file.
    - Execute in the command line: Run 
    ```your_SCRATCH_installation_path/bin/run_SCRATCH-1D_predictors.sh 	test_seq.fasta test_seq 8``` ```8``` is the number of processors, ```test_seq``` is the output files' prefix.
    - It will return four files in current folder:
        * test_seq.ss
        * test_seq.ss8
        * test_seq.acc
        * test_seq.acc20
* Calculate features for test sequences.
    - Execute in the command line: Run
    ```R --vanilla < PaRSnIP.R test_seq.fasta test_seq.ss test_seq.ss8 test_seq.acc20 test_seq```
    - After this step, one file will be created:
        * test_seq_src_bio: contains biological features corresponding to the raw protein sequences
* Use AlphaFold<sup><a href="#ref3">[3]</a></sup> or ColabFold<sup><a href="#ref4">[4]</a></sup> to get test sequences' pdb file
* Use Foldseek<sup><a href="#ref5">[5]</a></sup> to get the test sequences' 3di file
* Run get_3di.py to get input sequence
* Replace the parameters in ```bash_infer.sh```and run the script ```Bash bash_infer.sh``` to infer the test sequences result, or replace the parameters in ```bash_s2asolp.sh``` and run the script ```Bash bash_s2asolp.sh``` to retrain the model. 

1. <p name = "ref1">Magnan C N, Baldi P. SSpro/ACCpro 5: almost perfect prediction of protein secondary structure and relative solvent accessibility using profiles, machine learning and structural similarity[J]. Bioinformatics, 2014, 30(18): 2592-2597.</p>
2. <p name = "ref2">Su J, Han C, Zhou Y, et al. SaProt: protein language modeling with structure-aware vocabulary[J]. bioRxiv, 2023: 2023.10. 01.560349.</p>
3. <p name = "ref3">Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold[J]. Nature, 2021, 596(7873): 583-589.</p>
4. <p name = "ref4">Mirdita M, Schütze K, Moriwaki Y, et al. ColabFold: making protein folding accessible to all[J]. Nature methods, 2022, 19(6): 679-682.</p>
5. <p name = "ref5">Van Kempen M, Kim S S, Tumescheit C, et al. Fast and accurate protein structure search with Foldseek[J]. Nature Biotechnology, 2023: 1-4.</p>