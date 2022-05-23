1. Folder Description
  This is TripleBrain written in C# and Matlab, divided into two folders:
  "Fixed_Point": an algorithm based on floating point implementation;
  "FloatPoint":   an algorithm based on fixed point implementation;
  "Image_Proc_Code": Image (including static image and DVS dataset) preprocessing script, including static image downsampling, DOG                                   filtering and AER preprocessing of DVS dataset

  In order to conduct a more direct test on the 8 datasets of MNIST, ETH-80, ORL-10, ORL-10-ext, Yale-10, N-MNIST, Poker-DVS also called Cards-DVS, Posture-DVS, at the same time, to display its parameters more intuitively, we respectively include 8 folders in "Fixed_Point" and "FloatPoint".
  Under the "Fixed_Point" folder, there are 8 folders as follows:
    1) TripleBrain_MNIST
    2) TripleBrain_ETH_80
    3) TripleBrain_ORL
    4) TripleBrain_ORL-10-ext
    5) TripleBrain_Yale-10
    6) TripleBrain_NMNIST
    7) TripleBrain_PokerDVS
    8) TripleBrain_PostureDVS

  It should be noted that these 8 folders contain two algorithms, SOM-STDP&R-STDP and R-SOM-STDP, and each algorithm contains two modes: all-pair model and nearest-pair mode. So each of these 8 folders contains 4 subfolders for algorithms and two other subfolders for intermediate and result data storage. Since the directory structure of each subfolder is the same, the following is a detailed description of Fixed_Point\TripleBrain_MNIST.

The Fixed_Point\TripleBrain_MNIST folder contains 6 folders, as follows:
 Two algorithms and two modes
    1) 1_all-pair SOM-STDP&R-STDP
    2) 2_nearest-pair SOM-STDP&R-STDP
    3) 3_all-pair R-SOM-STDP
    4) 4_nearest-pair R-SOM-STDP
 Intermediate data and results
    5) DataSet
    6) Result

    Folders 1) to 4): describes the TripleBrain algorithm, including two modes (all-pair and nearest-pair) configurable SOM-STDP & R-STDP and R-SOM-STDP. The related files are described as follows:
    Parameter.cs:  
    parameter file, which contains parameters that can be defined or configured.

    PoisionCoding.cs:  
    coding file, used for pulse coding of static images, supports three coding methods (Poisson coding, temporal coding and ISI).

    SOM-RSTDP_Model.cs: 
    The algorithm part, including training and labeling neurons (if needed) and testing, note that there are also some parameters that need to      be changed in this file.

    Program.cs: 
    main function entry.

  "DataSet": Describes the processed DVS dataset or static image and its encoding file, the visualization code file "WeightVisable.m" written by matlab and the visualized image "Total.bmp", other related training obtained model parameters and prediction files, etc.

  "Result": Stores the final recognition accuracy of two algorithms and two modes.

2. Tools
visual studio 2019 and above (preferably 2019)
matlab

3. Pay attention
  The algorithm code can be directly run and the result can be obtained. The algorithm parameters are not optimal parameters, and the recognition accuracy can be further adjusted by adjusting the parameters.

