# Introduction

This program implements a deep learning framework focused on using Convolutional Neural Networks (CNN) to predict phenotypic traits in plants or animals from genomic data. It aims to enhance the accuracy and efficiency of genomic selection by analyzing genomic data to predict phenotypic traits.
* DCNGP Model: https://github.com/shuaizengMU/G2PDeep_model](https://github.com/xiangweidai/DCNGP

## Data Preparation
This section outlines the steps for preparing genomic data for analysis. We focus on dimensionality reduction of the input VCF files through PCA (Principal Component Analysis). Two methods are presented: using Plink for PCA, and transforming VCF files into a 012 matrix followed by PCA using Python libraries.
Method 1: PCA with Plink
This section describes a detailed procedure for performing PCA on VCF files using Plink. The process involves retaining a number of principal components equal to the number of samples initially, then determining the number of principal components to retain based on the proportion of variance explained by each component.
1. **Plink Installation**:
   Ensure Plink (preferably version 2.0 or later) is installed on your system. Plink can be downloaded from [the Plink website](https://www.cog-genomics.org/plink/).
2. **Initial PCA with Plink**:
   Run PCA on your VCF file using Plink with the following command:
   plink2 --vcf path_to_your_data.vcf --pca number_of_samples --out pca_output
   - `--vcf`: Specifies the path to your VCF file.
   - `--pca number_of_samples`: Replace `number_of_samples` with the actual number of samples in your dataset. This will initially compute as many principal components as there are samples.
   - `--out`: Specifies the output file prefix.
3. **Analyzing Output**:
   Plink will produce output files including:
   - `pca_output.eigenval`: Lists the eigenvalues (variance explained) for each principal component.
   - `pca_output.eigenvec`: Contains the actual principal components for each sample.
4. **Determining Principal Components to Retain**:
   - Open the `pca_output.eigenval` file and calculate the proportion of variance explained by each principal component.
   - A common approach is to retain components that explain a significant amount of variance (e.g., using a cumulative variance threshold like 95%).
   - The formula for the proportion of variance explained by each PC is:
     Proportion of Variance = Eigenvalue of PC / Sum of all Eigenvalues
   - Add these proportions cumulatively until your desired threshold of total variance explained is reached.
5. **Final PCA with Selected Components**:
   - Once you've determined how many principal components to retain (say `N` components), run Plink again with this number:
     plink2 --vcf path_to_your_data.vcf --pca N --out final_pca_output
   - This will give you the final PCA output with the desired number of components.
By following these steps, you will be able to perform PCA using Plink effectively, retaining the most informative principal components for your genomic data.
Method 2: PCA on 012 Matrix with Python
Conversion to 012 Matrix:
First, convert the VCF file to a 012 matrix (0 for homozygous reference, 1 for heterozygous, and 2 for homozygous variant).
This can be done using various tools or scripts that parse VCF files and output the genotype in the 012 format.
Python PCA:
Import the 012 matrix into a Python environment.
Utilize libraries like scikit-learn to perform PCA.
Importing Principal Components into a CSV File:
1. Import the principal component data obtained from the PCA analysis into a new CSV file.
2. Insert the corresponding phenotypic data into the first column of the CSV file.
3. Randomly divide the training and test sets:
   Use a 4:1 ratio to randomly divide the data into training and test sets. Save the divided training and test sets as two separate CSV files. The file names should include the name of the phenotype and end with .train.csv and .test.csv as suffixes, for example, phenotype_name.train.csv and phenotype_name.test.csv.

## Install
Python 3.6.8
```
pip install -r requirement.txt
```
## Running the program

```
1、Training the Model
./main --data_dir ./data --result_dir ./result --dataset_type gal  --batch_size 16 --early_stopping_patience 16 --reduce_lr_patience 10 --leaky_alpha 0.7  --learning_rate 0.0001 --scaler_path ./result/scaler.joblib
--data_dir ./data           # Directory of the dataset
--result_dir ./result       # Directory for output results
--dataset_type twg          # Type of dataset
--batch_size 16             # Batch size for training
--early_stopping_patience 16# Patience for early stopping
--reduce_lr_patience 10     # Patience for reducing learning rate
--learning_rate 0.0001      # Initial learning rate
--scaler_path ./result/scaler.joblib # Path to save the StandardScaler state
After the training is complete, the model's prediction results and performance evaluations will be saved in the result_dir directory you specified. You can view the following files:
pcc_twg.csv: Contains the model's predictions for the test set.
twg_model.h5: The trained model file.
2、Prediction
./prediction --model_path ./result/gal_model.h5 --data_path ./data/gal.csv --scaler_path ./result/scaler.joblib --output_path ./result
```

## License

[Apache License 2.0](LICENSE)
