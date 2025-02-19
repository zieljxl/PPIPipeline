#PPI-BI

This pipeline is developed for sequence-based protein-protein interaction and binding interface prediction using deep learning methods devloped by different researchers. 
    Dependencies and packages needed:
    Python 3.5.2
    Numpy 1.14.1
    Gensim 3.4.0
    HDF5 and h5py
    Pickle
    Scikit-learn 0.19
    Tensorflow 1.2.0
    Keras 1.2.0 (for DeepFE-PPI) Keras 2.1.2(for DeePPI)
    Numba
    Pytorch
    Terminal
    Hmmer (Please set the correct path for hmmer bin inside the python code)
    DeepFE-PPI (https://github.com/xal2019/DeepFE-PPI.git)
    DeepPPI (gitlab.univ-nantes.fr:richoux-f/DeepPPI.git)
    ResPRE (https://github.com/leeyang/ResPRE.git)
The copyright of original DeepFE-PPI, DeepPPI and ResPRE scripts belong to their own authors. 
# Scientific Aim 1: Protein-protein interaction prediction
  Please follow the dataset examples to prepare your own datasets.
  - Want to use DeepFE-PPI method. Please put the code under the DeepFE-PPI folder.
    ```sh
    python3 ppi_prediction.py protein1 protein2
    ```
  - Want to use DeepPPI method. Please put the code and model under the DeepPPI-keras folder.
    ```sh
    python3 ppi_prediction_planb.py protein1 protein2 whether_remove_polyq
    ```
  - Protein 1/2 (Uniprot Entry, such P10275)
  - For DeepPPI, you can also decide to remove polyq aggregates or not
  
You can also:
  - Test binding probabilities for multiple proteins. Please put the code under the DeepPPI-keras folder.
     ```sh
    python3 multi_body.py
    ```
  - You need to change the lsit of proteins inside the code
# Scientific Aim 2: Binding interface prediction
### Step I: Generating multiple sequence alignment. 
Please follow the steps indicated. 
Please make sure you have get_alignment bash script in the same directory.
Please download protein sequence dataset from Uniprot.
```sh
python3 hmm_seq.py
python3 Map_MSA.py MSA_file_1 MSA_file_2 Outputfile
```
### Step II: Calculate contact score for pairwise residues. Please use the ResPRE package from github
### Step III: Align multiple sequence alignment positions to structural positions 
```sh
python3 align_hmm_with_pdb.py hmm_profile_file fasta_sequence_file output_scan_file
```
### Step IV: Extract contacts from crystal structure if any
```sh
python3 interface_contact_calpha.py pdb_file 1st_chain 2nd_chain distance_cutoff
```
### StepV: Compare predicted contacts with real contacts
```sh
python3 process_res_pairs.py score_file(from step II) length_of_1st_protein align_file_protein1 (from step III) align_file_protein2(from step III) contact_file(from step IV) number_of_contacts_to_plot svaed_score_file
```
