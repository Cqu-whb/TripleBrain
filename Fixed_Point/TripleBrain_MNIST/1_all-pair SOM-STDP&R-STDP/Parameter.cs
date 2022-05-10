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

        private const int ImageColumnSize = 28;      //图像宽 mnist:28 orl:32 yale:32 eth80:32
        private const int ImageRowSize = 28;        //图像高 mnist:28 orl:32 yale:32 eth80:32

        public static int InputLayerNeurons = ImageColumnSize * ImageRowSize;  //输入神经元的个数（按一维排列）
        public static int NumClass = 10;          //类别数 mnist:10 orl:10 yale:10 eth80:8 posture-dvs:3 cards-dvs:4
        public const  int SOM_ROW = 16;        //SOM二维阵列的高
        public const  int SOM_COL = 16;        //SOM二维阵列的宽
        public static  int[,] SOM_Maps = new int[SOM_ROW, SOM_COL];   //SOM二维阵列
        public const int OutputLayerNeurons = SOM_ROW * SOM_COL;      //输出神经元
        public static int[,] WeightTensor = new int[OutputLayerNeurons, InputLayerNeurons];  //权重矩阵
        public static int[] PreExpTable = new int[SampleTimeWindow];
        public static int[] PostExpTable = new int[SampleTimeWindow];
        public static int[] VmExpTable = new int[SampleTimeWindow];
        public static int WeightUpperBound = (16 << FIXED_POINT_12) - 1;
        public static int WeightLowerBound = 0 << FIXED_POINT_12;

        //som-stdp&rstdp: 1.0  som-r-stdp:  5.0 
        public static int MembraneThreshold = (int)Math.Round(1.5 * (1 << FIXED_POINT_8)); 
        public static int RestPotential = (int)Math.Round(0.0 * (1 << FIXED_POINT_8));
        public const int RefractionPeriod = 0;
        public const int SampleTimeWindow = 200;
        //som-stdp&rstdp : 200  som-r-stdp: 200 
        public static double V_tao = 200;
        //som-stdp&rstdp: 200   som-r-stdp:200 
        public static double DynamicThreshold_tao = 200; 
        //可以定义浮点的1 /tao
        public static int[] DynamicThreshold = new int[OutputLayerNeurons];       //动态阈值
        //som-stdp&rstdp mnist: 5.0   som-r-stdp: 4.0
        public static int DynamicThreasholdAmount = (int)Math.Round(5.0 * (1 << FIXED_POINT_8));
        //NeuronSTDPTrace的各项参数  
        //som-stdp&rstdp mnist: 16 orl: 16 yale:16 eth80:150  som-r-stdp: orl:16 yale:16 eth80:50
        public static double PreSpike_tao  =  16;  //nearest model: 100
        //som-stdp&rstdp : 16  som-r-stdp:  20
        public static double PostSpike_tao =  20;  //nearest model:100
        public static int PreTraceAmountChange = (int)Math.Round(1.0 * (1 << FIXED_POINT_4));
        public static int PostTraceAmountChange = (int)Math.Round(1.0 * (1 << FIXED_POINT_4));

        //学习率
        public static int PositiveLR = (int)Math.Round(0.01  * (1 << FIXED_POINT_10)); 
        public static int NegativeLR = (int)Math.Round(0.001 * (1 << FIXED_POINT_10)); 
        
        // 指定标签所需变量
        public static int[,] LabelNeuronResponse = new int[OutputLayerNeurons, NumClass]; // 记录每个输出神经元对于标签的响应情况


        // 神经元对应标签
        public static int[] NeuronRelativeLabel = new int[OutputLayerNeurons]; // 记录每个输出神经元对应的标签
        public static int[] EachLabelNeuronNum = new int[NumClass];  //每个标签所占输出神经元的个数
        public static int NumCorrect = 0;
        //SOM相关变量
        public static int inhibitorFactor = (int)Math.Round(0.125 * (1 << FIXED_POINT_4)); //mnist 0.125 orl 0.125
        
        //相关数据统计
        public static int[] LabelCorrectNum = new int[NumClass];    //用于统计每个类别识别样本正确的个数
        public static int[,] truelabel2predictlabel = new int[NumClass, NumClass];
        public static int[] EachLabelNum = new int[NumClass];
        public static int[] nofireimagelabel = new int[NumClass];

        public static int[] image_for_category_fire_num = new int[NumClass];
        public static ArrayList category3_wrong_index = new ArrayList();
    }
}