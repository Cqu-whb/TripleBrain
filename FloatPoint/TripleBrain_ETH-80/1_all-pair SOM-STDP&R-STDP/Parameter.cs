using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;
using System.Linq;
using System.IO;
using SOM_RSTDPModel;

namespace PARAMETER
{
    class Parameter
    {
        //定点数设置
        public const int FIXED_POINT_12 = 12; //12位定点数
        public const int FIXED_POINT_8 = 8; //8位定点数
        public const int FIXED_POINT_4 = 4;  //4位定点数
        public const int FIXED_POINT_10 = 10;  //10位定点数

        private const int ImageColumnSize = 32;      //图像宽 mnist:28 orl:32 yale:32 eth80:32
        private const int ImageRowSize = 32;        //图像高 mnist:28 orl:32 yale:32 eth80:32

        public static int InputLayerNeurons = ImageColumnSize * ImageRowSize;  //输入神经元的个数（按一维排列）
        public static int NumClass = 8;          //类别数 mnist:10 orl:10 yale:10 eth80:8 posture-dvs:3 cards-dvs:4
        public const  int SOM_ROW = 16;        //SOM二维阵列的高
        public const  int SOM_COL = 16;        //SOM二维阵列的宽
        public static  int[,] SOM_Maps = new int[SOM_ROW, SOM_COL];   //SOM二维阵列
        public const int OutputLayerNeurons = SOM_ROW * SOM_COL;      //输出神经元
        public static double[,] WeightTensor = new double[OutputLayerNeurons, InputLayerNeurons];  //权重矩阵
        public static double[] PreExpTable = new double[SampleTimeWindow];
        public static double[] PostExpTable = new double[SampleTimeWindow];
        public static double[] VmExpTable = new double[SampleTimeWindow];
        public static double WeightUpperBound = 16.0;
        public static double WeightLowerBound = 0.0;
        /*--------------------MembraneThreshold Config---------------------------------------/
        all-pair model  
            som-stdp&rstdp: eth80: 1.5 
            som-r-stdp:     eth80: 0.1
        nearest-pair model
            som-stdp&rstdp: eth80: 1.5
            som-r-stdp:     eth80: 0.1 
        /--------------------MembraneThreshold Config---------------------------------------*/
        public static double MembraneThreshold = 1.45; 
        public static double RestPotential = 0.0;
        public const int RefractionPeriod = 0;
        public const int SampleTimeWindow = 200;
        /*--------------------V_tao Config---------------------------------------/
        all-pair model  
            som-stdp&rstdp:  eth80: 200 
            som-r-stdp:      eth80: 200 
        nearest-pair model
            som-stdp&rstdp:  eth80: 200 
            som-r-stdp:      eth80: 200 
        /--------------------V_tao Config---------------------------------------*/
        public static double V_tao = 200;
        /*--------------------DynamicThreshold_tao Config---------------------------------------/
        all-pair model  
            som-stdp&rstdp: eth80:20000
            som-r-stdp:     eth80:20000 
        nearest-pair model
            som-stdp&rstdp: eth80:20000 
            som-r-stdp:     eth80:20000 
        /--------------------DynamicThreshold_tao Config---------------------------------------*/
        public static double DynamicThreshold_tao = 20000; 
        //可以定义浮点的1 /tao
        public static double[] DynamicThreshold = new double[OutputLayerNeurons];       //动态阈值
        /*--------------------DynamicThreasholdAmount Config---------------------------------------/
        all-pair model  
            som-stdp&rstdp:  eth80:16.0 
            som-r-stdp:      eth80:0.5
        nearest-pair model
            som-stdp&rstdp:  eth80:16.0 
            som-r-stdp:      eth80:0.5
        /--------------------DynamicThreasholdAmount Config---------------------------------------*/
        public static double DynamicThreasholdAmount = 16.0;
        //NeuronSTDPTrace的各项参数  
        /*--------------------PreSpike_tao Config---------------------------------------/
        all-pair model  
            som-stdp&rstdp:  eth80:150 
            som-r-stdp:      eth80:50 
        nearest-pair model
            som-stdp&rstdp:  eth80:150 
            som-r-stdp:      eth80:50
        /--------------------PreSpike_tao Config---------------------------------------*/
        public static double PreSpike_tao  =  160;
        /*--------------------PostSpike_tao Config---------------------------------------/
        all-pair model  
            som-stdp&rstdp: eth80:10000 
            som-r-stdp:     eth80:50 
        nearest-pair model
            som-stdp&rstdp: eth80:10000
            som-r-stdp:     eth80:50 
        /--------------------PostSpike_tao Config---------------------------------------*/
        public static double PostSpike_tao =  20000; 
        public static double PreTraceAmountChange = 1.0;
        public static double PostTraceAmountChange = 1.0;

        //学习率
        public static double PositiveLR = 0.01; 
        public static double NegativeLR = 0.001; 
        
        // 指定标签所需变量
        public static double[,] LabelNeuronResponse = new double[OutputLayerNeurons, NumClass]; // 记录每个输出神经元对于标签的响应情况


        // 神经元对应标签
        public static int[] NeuronRelativeLabel = new int[OutputLayerNeurons]; // 记录每个输出神经元对应的标签
        public static int[] EachLabelNeuronNum = new int[NumClass];  //每个标签所占输出神经元的个数
        public static int NumCorrect = 0;
        //SOM相关变量
        public static double inhibitorFactor = 0.125; 
        
        //相关数据统计
        public static int[] LabelCorrectNum = new int[NumClass];    //用于统计每个类别识别样本正确的个数
        public static int[,] truelabel2predictlabel = new int[NumClass, NumClass];
        public static int[] EachLabelNum = new int[NumClass];
        public static int[] nofireimagelabel = new int[NumClass];

        public static int[] image_for_category_fire_num = new int[NumClass];
        public static ArrayList category3_wrong_index = new ArrayList();
    }
}