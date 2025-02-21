### mlstabilitytest

mlstabilitytest is a package that facilitates testing machine learning models for formation energy on stability predictions.

This package was created in conjunction with [this manuscript](https://www.nature.com/articles/s41524-020-00362-y).

If you use this package, please cite
```
Bartel, C., Trewartha, A., Wang, Q., Dunn, A., Jain, A., Ceder, G., 
A critical examination of compound stability predictions from machine-learned formation energies, 
npj Computational Materials 6, 97 (2020)
```

The source code for this repository is available at https://github.com/CJBartel/TestStabilityML.

This package serves two purposes:
1. Reproduce the results described in the aforementioned paper.
2. Allow the community to quickly repeat this analysis for newly developed models.

### Installation

You can install mlstabilitytest by cloning the repository:
```
git clone https://github.com/CJBartel/TestStabilityML.git
```
Then navigating to your cloned directory and installing:
```
cd TestStabilityML
python setup.py install --user
```

### Reproducing published results
The data used for training and testing each model was extracted from the Materials Project (MP) and is stored for convenience as a set of .json files in:
```
mlstabilitytest/mp_data/data/
```
* Ef.json is the original MP data of ground-state formation energies for all non-elemental compositions.
* Other files are stored to make running the stability analysis faster.

Classes that allow for re-training of the examined ML models are available within: 
```
mlstabilitytest/training/
```
* CGCNN is not included in this framework because training requires the storage of a very large file (all ground-state structures in Materials Project).
* Roost is not included in this framework because it was implemented exactly as provided in https://github.com/CompRhys/roost.

An example script that would re-train all models is provided at:
```
mlstabilitytest/train_models.py
```
Classes that allow for performing the stability analyses with the learned formation energies are provided at:
```
mlstabilitytest/stability/
```
Inputs (i.e., predicted energies) and outputs (i.e., resulting stabilities) are provided for each model as follows:
```
mlstabilitytest/ml_data/TRAINED_ON/EXPERIMENT/MODEL/
```
* TRAINED_ON 
    * Ef &rarr; ML models trained on formation energies. 
    * Ed &rarr; ML models trained on decomposition energies.
* EXPERIMENT
    * allMP &rarr; ML models are trained and evaluated on all of Materials Project.
    * LiMnTMO &rarr; ML models are trained on allMP minus quaternary Li-Mn-TM-O compounds and evaluated on these excluded compounds.
    * smact &rarr; ML models are trained on allMP minus quaternary Li-Mn-TM-O compounds and evaluated on a large list of candidate formulas in this chemical space generated using https://github.com/WMD-group/SMACT.
    * classifier &rarr; ML models trained on classifying compounds as stable or unstable
    * random &rarr; random error baselines for each ML model for assessment of error cancellation
* MODEL
    * This can be any of the 7 models studied in this work.
* within each directory, ml_input.json is a dictionary containing formulas and their ML-predicted properties and ml_results.json is a dictionary with the resulting stability analysis for each formula.

An example script that would repeat all stability analyses is available at:
```
mlstabilitytest/analyze_models.py
```
All Figures and Tables shown in the manuscript can be re-generated from the provided data using:
```
mlstabilitytest/plot.py
```

### Repeating this analysis for a new model
Performing the same stability analyses with new predicted formation energies will follow the example provided in:
```
mlstabilitytest/analyze_models.py
```
1. Produce ml_input.json using cross-validation
    * {formula (str) : predicted formation energy (float, eV/atom) for formulas relevant to particular test}
        * each experiment (allMP, LiMnTMO, smact, etc.) requires a certain set of formulas which can be obtained by comparing with mlstabilitytest/ml_data/TRAINED_ON/EXPERIMENT/MODEL/ml_input.json
2. Evaluate how well the learned formation energies do on each experiment (allMP, LiMnTMO, smact) utilizing:
    ```
    mlstability.stability.StabilityAnalysis.StabilityAnalysis 
    ```

### Fluidity of Materials Project
Please note that the Materials Project database is constantly changing. While this doesn't present any issues for the direct replication of our results or the application of new models trained on formation energy, it may complicate the strict replication of our results for models trained on multiple properties (e.g., band gap and formation energy learned simultaneously).

#### Please feel free to post an issue or contact cbartel [at] berkeley [dot] edu for assistance (email will be faster)