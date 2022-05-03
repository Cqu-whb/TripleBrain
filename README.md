# TripleBrain
Triplebrain fixed-point and floating-point algorithms


1. Folder Description
This is TripleBrain written in C#, divided into two folders：
"Fixed_Point": an algorithm based on floating point implementation;
"FloatPoint":   an algorithm based on fixed point implementation.

 "Fixed_Point" and "FloatPoint" respectively contain 7 dataset-related algorithms, each of which contains "DataSet" and "SOM_RSTDP".

 "DataSet": Describes the processed DVS dataset or static image and its encoding file, the visualization code file "WeightVisable.m" written by matlab and the  visualized image "Total.bmp", other related training obtained model parameters and prediction files, etc.

 "SOM_RSTDP": The folder describes the TripleBrain algorithm, including two modes (all-pair and nearest-pair) configurable SOM-STDP & R-STDP and R-SOM-STDP. The related files are described as follows:

 Parameter.cs:  
 parameter file, which contains parameters that can be defined or configured.
 PoisionCoding.cs:  
 coding file, used for pulse coding of static images, supports three coding methods (Poisson coding, temporal coding and ISI).
 SOM-RSTDP_Model.cs: 
  The algorithm part, including training and labeling neurons (if needed) and testing, note that there are also some parameters that need to be changed in this file.
 Program.cs: 
 main function entry.

2. Tools
visual studio 2019 and above (preferably 2019)
matlab

3. operate  
After modifying the relevant parameters, you can directly run


4. Pay attention
Due to the modification of some parameters, the recognition accuracy of some DataSet is slightly different from the data of the paper. Please refer to the latest parameters.
