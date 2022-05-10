using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;
using System.Linq;
using System.IO;
using PARAMETER;
using System.Threading.Tasks;
using System.Diagnostics;


namespace SOM_RSTDPModel
{
    public static class Som_rstdpModel
    {
        /*--------------------r-stdp stage learning epoch ---------------------------------------/
        all-pair model  
            som-stdp&rstdp:   yale: 15
            som-r-stdp:       yale: 15
        nearest-pair model
            som-stdp&rstdp:   yale: 20
            som-r-stdp:       yale: 15
        /--------------------r-stdp stage learning epoch---------------------------------------*/
        public static int RSTDP_EP = 20;

        public const string Path = "../../../../DataSet/";
        public const string LabelPath = Path + "labelPlot.txt";
        public const string WeightPath = Path + "weight9.txt";
        public const string DynamicPath = Path + "DynamicThreshold.txt";
        public const string EachLabelNeuronNumPath = Path + "EachLabelNeuronNum.txt";
        //public static string RSTDP_WEIGHT_PATH = Path + "RSTDPweight" + (RSTDP_EP-1).ToString() + ".txt";
        public static string RSTDP_WEIGHT_PATH = Path + "RSTDPweight" + (RSTDP_EP - 1).ToString() + ".txt";
        public const string RSTDP_DynamicPath = Path + "strp_DynamicThreshold.txt";
        public static double GlobalCorrectRate = 0.0;


        public const string software_weight0_path = "E:/shujuduibi/software_weight0.txt";
        public const string software_weight1_path = "E:/shujuduibi/software_weight1.txt";
        public const string software_weight2_path = "E:/shujuduibi/software_weight2.txt";
        public const string software_weight3_path = "E:/shujuduibi/software_weight3.txt";
        public const string sorftware_pretrace_path = "E:/shujuduibi/pretrace.txt";
        

        //用于modelsim仿真
        public const string InitWeightPath = Path + "init_weight.txt";
        public const string PreTrace_EXP_LUT_Path = Path + "pretrace_exp_lut.txt";
        public const string PostTrace_EXP_LUT_Path = Path + "posttrace_exp_lut.txt";
        public const string Vm_EXP_LUT_Path = Path + "vm_exp_lut.txt";

        //yale
        public const string TrainCodingSpikePath = Path + "yale_face_train_coding.txt";
        public const string TrainLabelPath = Path + "yale_face_train_label.txt";
        public const string TestCodingSpikePath = Path + "yale_face_test_coding.txt";
        public const string TestLabelPath = Path + "yale_face_test_label.txt";
        public static int ImageOffest = 38;
        public static int TrainNum = 38;
        public static int TestNum = 257;
        public static int MNISTTrainNum = 382;
        public static int MNISTTestNum = 257;
        public static int MAX_EPOCH = 10;
    
        //Parameter
        public static int[][] TrainCodingSpike = new int[MNISTTrainNum][];
        public static int[][] TestCodingSpike = new int[MNISTTestNum][];
        public static int[] TrainLabel = new int[MNISTTrainNum];
        public static int[] TestLabel = new int[MNISTTestNum];
        public static Random  WeightRandomHandle = new Random(0);
        
        //public static double WeightFactor = 0.1;
        public static double WeightFactor = 0.1;
        public static int EP = 0;
        


        //记录标签过程中没有发射脉冲的图像数
        public static int nofire_imagenum = 0;
        //记录inference时每幅图像在神经元发射的脉冲数
        public static int[] imagefirenum= new int[Parameter.OutputLayerNeurons];
        //预测
        public static int preLabel = -1;

        public static int imgNum = 0;
        //文件读取句柄（ReaderStream）
        static FileStream fs_train_data = new FileStream(TrainCodingSpikePath, FileMode.Open);    //打开训练集文件
        static StreamReader sr_train_data = new StreamReader(fs_train_data);
        static FileStream fs_train_label = new FileStream(TrainLabelPath, FileMode.Open);
        static StreamReader sr_train_label = new StreamReader(fs_train_label);
        static FileStream fs_test_data = new FileStream(TestCodingSpikePath, FileMode.Open);    //打开测试集文件
        static StreamReader sr_test_data = new StreamReader(fs_test_data);
        static FileStream fs_test_label = new FileStream(TestLabelPath, FileMode.Open);
        static StreamReader sr_test_label = new StreamReader(fs_test_label);


        public static void FeedforwardPass(int[] ImageData,int InputClassLabel,string LearnRule,int img_num, string pattern)
        {
            //膜电位
            int[] NeuronMembranePotential = new int[Parameter.OutputLayerNeurons]; //膜电位
            bool InhibitoryTimes; //每个时间步下是否有脉冲输出标志位
            int[] SingleStreamResponse = new int[Parameter.OutputLayerNeurons];  //单张测试图像的神经元响应
            int[] PreSpikeTrace = new int[Parameter.InputLayerNeurons];
            int[] PostSpikeTrace = new int[Parameter.OutputLayerNeurons];
            int [] PreSpikeTraceLastUpdateTime = new int[Parameter.InputLayerNeurons]; //记录上一次PreTrace更新的时刻
            int[]  PostSpikeTraceLastUpdateTime = new int[Parameter.OutputLayerNeurons]; //记录上一次PostTrace更新的时刻
            int preTime = ImageData[0];
            ArrayList SOMSpikeQue = new ArrayList();

            ArrayList SOMSpikeQue0 = new ArrayList();
            ArrayList SOMSpikeQue1 = new ArrayList();
            ArrayList SOMSpikeQue2 = new ArrayList();
            ArrayList SOMSpikeQue3 = new ArrayList();

            bool [] output_fire_flag = new bool[Parameter.OutputLayerNeurons];

            for (int EvenSpikeNum = 0; EvenSpikeNum <= (int)(ImageData.Length / 2); EvenSpikeNum++)  //每张图像事件的个数
            {
                InhibitoryTimes = false;
                //判断当前事件的时刻与上一事件时刻是否相同（当事件同一时刻下时先处理完同一时刻下的所有脉冲（事件）时再进行突触后脉冲的判断）
                if(EvenSpikeNum == (int)(ImageData.Length / 2) || ImageData[2 * EvenSpikeNum] != preTime)
                {
                    //(1) 检索output_neuron_fire_flag[256]，进行post-spike的权重更新
          
                    for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
                    {
                        if (output_fire_flag[i] == true)
                        {

                            //更新post trace
                            int PostTraceDffTime = Math.Abs(preTime - PostSpikeTraceLastUpdateTime[i]);
                            
                            //nearest
                            PostSpikeTrace[i] = Parameter.PostTraceAmountChange;

                            //all pair 
                            //PostSpikeTrace[i] = ((Parameter.PostExpTable[PostTraceDffTime] * PostSpikeTrace[i]) >> Parameter.FIXED_POINT_4) + Parameter.PostTraceAmountChange;
                            PostSpikeTraceLastUpdateTime[i] = preTime;

                            //SOMSpikeQue.Add(i);
                            InhibitoryTimes = true;
                            if (LearnRule == "SOM")  //所有突触后脉冲的学习都放到一个模块里面，直到fifo为空且神经元下标为63时done信号生效
                            {
                                for (int j = 0; j < Parameter.InputLayerNeurons; j++)
                                {
                                    int delta_time = preTime - PreSpikeTraceLastUpdateTime[j];
                                    Parameter.WeightTensor[i, j] += (((Parameter.PositiveLR * ((PreSpikeTrace[j] * Parameter.PreExpTable[delta_time]) >> 4 << 6)) >> 10) << 2);
                                    if (Parameter.WeightTensor[i, j] > Parameter.WeightUpperBound)
                                        Parameter.WeightTensor[i, j] = Parameter.WeightUpperBound;
                                    else if (Parameter.WeightTensor[i, j] < Parameter.WeightLowerBound)
                                        Parameter.WeightTensor[i, j] = Parameter.WeightLowerBound;
                                }
                            }
                            else if (LearnRule == "R_STDP")
                            {


                                for (int j = 0; j < Parameter.InputLayerNeurons; j++)
                                {
                                    int delta_time = preTime - PreSpikeTraceLastUpdateTime[j];
                                    if (InputClassLabel == Parameter.NeuronRelativeLabel[i])
                                    {
                                        /*--------------------post learning reward lr factor---------------------------------------/
                                        all-pair model  
                                            som-stdp&rstdp: yale:1.0
                                            som-r-stdp:     yale:1.0
                                        nearest-pair model
                                            som-stdp&rstdp: yale:1.0
                                            som-r-stdp:     yale:1.0
                                        /--------------------post learning reward lr factor---------------------------------------*/
                                        Parameter.WeightTensor[i, j] += (((((int)Math.Round(1.0 * (1 << 10)) * Parameter.PositiveLR) >> 10) * ((PreSpikeTrace[j] * Parameter.PreExpTable[delta_time]) >> 4 << 6) >> 10) << 2);
                                    }
                                    else
                                    {
                                        /*--------------------post learning punish lr factor---------------------------------------/
                                        all-pair model  
                                            som-stdp&rstdp: yale:8.25
                                            som-r-stdp:     yale:1.0
                                        nearest-pair model
                                            som-stdp&rstdp: yale:8.25
                                            som-r-stdp:     yale:1.0
                                        /--------------------post learning punish lr factor--------------------------------------*/
                                        Parameter.WeightTensor[i, j] -= (((((int)Math.Round(8.25 *  (1 << 10)) * Parameter.PositiveLR) >> 10) * ((PreSpikeTrace[j] * Parameter.PreExpTable[delta_time]) >> 4 << 6) >> 10) << 2);
                                    }
                                    if (Parameter.WeightTensor[i, j] > Parameter.WeightUpperBound)
                                        Parameter.WeightTensor[i, j] = Parameter.WeightUpperBound;
                                    else if (Parameter.WeightTensor[i, j] < Parameter.WeightLowerBound)
                                        Parameter.WeightTensor[i, j] = Parameter.WeightLowerBound;
                                }
                            }

                        }
                    }
                    //(2) 检索output_neuron_fire_flag[256]，进行抑制
                    //
                    if (InhibitoryTimes && (LearnRule == "ClassAssign" || LearnRule == "Predict" || (LearnRule == "R_STDP" && pattern == "SOM-STDP&RSTDP")))
                    {
                        for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
                        {
                            if (NeuronMembranePotential[i] != Parameter.RestPotential)
                                NeuronMembranePotential[i] = Parameter.RestPotential;
                        }
                    }
                    else if (InhibitoryTimes && (LearnRule == "SOM" || pattern == "SOM-RSTDP"))
                    {

                        for (int i = 0; i < SOMSpikeQue0.Count; i++)
                        {
                            SOMSpikeQue.Add(Convert.ToInt32(SOMSpikeQue0[i]));
                        }
                        for (int i = 0; i < SOMSpikeQue1.Count; i++)
                        {
                            SOMSpikeQue.Add(Convert.ToInt32(SOMSpikeQue1[i]));
                        }
                        for (int i = 0; i < SOMSpikeQue2.Count; i++)
                        {
                            SOMSpikeQue.Add(Convert.ToInt32(SOMSpikeQue2[i]));
                        }
                        for (int i = 0; i < SOMSpikeQue3.Count; i++)
                        {
                            SOMSpikeQue.Add(Convert.ToInt32(SOMSpikeQue3[i]));
                        }
                        for (int i = 0; i < SOMSpikeQue.Count; i++)
                        {

                            //行
                            for (int k = 0; k < Parameter.SOM_ROW; k++)
                            {
                                //列
                                for (int m = 0; m < Parameter.SOM_COL; m++)
                                {
                                    if (NeuronMembranePotential[k * Parameter.SOM_ROW + m] != Parameter.RestPotential)
                                    {

                                        if (NeuronMembranePotential[k * Parameter.SOM_ROW + m] != Parameter.RestPotential)
                                        {

                                            int Dist = Math.Max(Math.Abs(Convert.ToInt32(SOMSpikeQue[i]) / Parameter.SOM_ROW - k), Math.Abs(Convert.ToInt32(SOMSpikeQue[i]) % Parameter.SOM_COL - m));
                                            //int Dist = Math.Abs(Convert.ToInt32(SOMSpikeQue[i]) / Parameter.SOM_ROW - k) +  Math.Abs(Convert.ToInt32(SOMSpikeQue[i]) % Parameter.SOM_COL - m);
                                            int InhibitorFactor = Dist * Parameter.inhibitorFactor * (EP + 1);
                                            if (InhibitorFactor > (1 << Parameter.FIXED_POINT_4))
                                                InhibitorFactor = (1 << Parameter.FIXED_POINT_4);
                                            NeuronMembranePotential[k * Parameter.SOM_ROW + m] = ((NeuronMembranePotential[k * Parameter.SOM_ROW + m] * (((1 << Parameter.FIXED_POINT_4) - InhibitorFactor) << Parameter.FIXED_POINT_4)) >> Parameter.FIXED_POINT_8);
                                        }
                                    }
                                }
                            }

                        }
                        SOMSpikeQue = new ArrayList();
                        SOMSpikeQue0 = new ArrayList();
                        SOMSpikeQue1 = new ArrayList();
                        SOMSpikeQue2 = new ArrayList();
                        SOMSpikeQue3 = new ArrayList();
                    }
                    //(3)output_neuron_fire_flag[256]清零

                    for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
                    {
                        output_fire_flag[i] = false;
                    }
                }
                /*************************************************************************************************************************************************************/
                //下一个事件的处理
                if (EvenSpikeNum < (int)(ImageData.Length / 2))
				{
                    //PreTrace的更新
                    int PreTraceDffTime = Math.Abs(ImageData[2 * EvenSpikeNum] - PreSpikeTraceLastUpdateTime[ImageData[2 * EvenSpikeNum + 1]]);
					
                    
                    //nearest 
                    PreSpikeTrace[ImageData[2 * EvenSpikeNum + 1]] = Parameter.PreTraceAmountChange;
                    //all pair
                    //PreSpikeTrace[ImageData[2 * EvenSpikeNum + 1]] = ((Parameter.PreExpTable[PreTraceDffTime] * PreSpikeTrace[ImageData[2 * EvenSpikeNum + 1]]) >> 4) + Parameter.PreTraceAmountChange;
                    PreSpikeTraceLastUpdateTime[ImageData[2 * EvenSpikeNum + 1]] = ImageData[2 * EvenSpikeNum];
					if (PreSpikeTrace[ImageData[2 * EvenSpikeNum + 1]] >= (1 << 8) - 1)
						PreSpikeTrace[ImageData[2 * EvenSpikeNum + 1]] = (1 << 8) - 1;

					//膜电位的更新
					for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
					{
						int VmDffTime = (ImageData[2 * EvenSpikeNum] - preTime);
                       
                        NeuronMembranePotential[i] = ((NeuronMembranePotential[i] * Parameter.VmExpTable[VmDffTime]) >> 8) + ((Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] + ((Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] & 8) << 1)) >> (Parameter.FIXED_POINT_12 - Parameter.FIXED_POINT_8));
                        //vth 与 vm的比较
                        if (output_fire_flag[i] == false && (NeuronMembranePotential[i] >= (Parameter.MembraneThreshold + Parameter.DynamicThreshold[i])))
                        {

                            output_fire_flag[i] = true;
                            //SOMSpikeQue.Add(i); //此处是否需要仲裁（仲裁信号），写入FIFO，一共8位，高6位表示神经元下标，低2位表示sub-network下标

                            if (i >= 0 && i < 64)
                            {
                                SOMSpikeQue0.Add(i);
                            }
                            else if (i >= 64 && i < 128)
                            {
                                SOMSpikeQue1.Add(i);
                            }
                            else if (i >= 128 && i < 192)
                            {
                                SOMSpikeQue2.Add(i);
                            }
                            else if (i >= 192 && i < 256)
                            {
                                SOMSpikeQue3.Add(i);
                            }
                           
                            if (LearnRule == "SOM")
                            {
                                //动态阈值
                                Parameter.DynamicThreshold[i] += Parameter.DynamicThreasholdAmount;
                                if (Parameter.DynamicThreshold[i] >= ((1 << 24) - 1))
                                    Parameter.DynamicThreshold[i] = ((1 << 24) - 1);
                            }
                            else if (LearnRule == "R_STDP")
                            {
                                //动态阈值
                                if (InputClassLabel == Parameter.NeuronRelativeLabel[i])
                                {
                                    Parameter.DynamicThreshold[i] += Parameter.DynamicThreasholdAmount;
                                    if (Parameter.DynamicThreshold[i] >= ((1 << 24) - Parameter.MembraneThreshold - 1))
                                        Parameter.DynamicThreshold[i] = ((1 << 24) - Parameter.MembraneThreshold - 1);
                                }
                            }
                            else if (LearnRule == "ClassAssign")
                            {
                                //训练输出脉冲 用于标签
                                Parameter.LabelNeuronResponse[i, InputClassLabel] += 1;
                            }
                            else if (LearnRule == "Predict")
                            {
                                //测试输出脉冲 用于测试
                                SingleStreamResponse[i] += 1;
                            } 
                        }
                        
                     
                        NeuronMembranePotential[i] = (output_fire_flag[i]) ? 0 : NeuronMembranePotential[i];
                        if (NeuronMembranePotential[i] >= ((1 << 24) - Parameter.MembraneThreshold - 1))  //加不加没关系
							NeuronMembranePotential[i] = ((1 << 24) - Parameter.MembraneThreshold - 1);
					}
					//当突触前脉冲到来时权重的更新
					if (LearnRule == "SOM")
					{
						for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
						{

                            int delta_time = ImageData[2 * EvenSpikeNum] - PostSpikeTraceLastUpdateTime[i];
							Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] -= (((Parameter.NegativeLR * ((PostSpikeTrace[i] * Parameter.PostExpTable[delta_time]) >> 4 << 6)) >> 10) << 2);
							if (Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] > Parameter.WeightUpperBound)
								Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] = Parameter.WeightUpperBound;
							else if (Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] < Parameter.WeightLowerBound)
								Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] = Parameter.WeightLowerBound;
                        }
					}
					else if (LearnRule == "R_STDP")
					{
						for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
						{
							int delta_time = ImageData[2 * EvenSpikeNum] - PostSpikeTraceLastUpdateTime[i];
                            /*--------------------pre learning reward lr factor---------------------------------------/
                            all-pair model  
                                som-stdp&rstdp:  yale:1.0
                                som-r-stdp:      yale:1.0
                            nearest-pair model
                                som-stdp&rstdp:  yale:1.0
                                som-r-stdp:      yale:1.0
                            /--------------------pre learning reward lr factor---------------------------------------*/
                            if (InputClassLabel == Parameter.NeuronRelativeLabel[i])  
                                Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] -= (((((((int)Math.Round(1.0 * (1 << 10))) * Parameter.NegativeLR) >> 10) * ((PostSpikeTrace[i] * Parameter.PostExpTable[delta_time]) >> 4  << 6)) >> 10) << 2);
                            /*--------------------pre learning punish lr factor---------------------------------------/
                            all-pair model  
                                som-stdp&rstdp:  yale:1.0
                                som-r-stdp:      yale:1.0
                            nearest-pair model
                                som-stdp&rstdp:  yale:1.0
                                som-r-stdp:      yale:1.0
                            /--------------------pre learning punish lr factor---------------------------------------*/
                            else
                                Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] += (((((((int)Math.Round(1.0 * (1 << 10))) * Parameter.NegativeLR) >> 10) * ((PostSpikeTrace[i] * Parameter.PostExpTable[delta_time]) >> 4 << 6)) >> 10) << 2);
							if (Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] > Parameter.WeightUpperBound)
								Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] = Parameter.WeightUpperBound;
							else if (Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] < Parameter.WeightLowerBound)
								Parameter.WeightTensor[i, ImageData[2 * EvenSpikeNum + 1]] = Parameter.WeightLowerBound;
						}
					}
                    preTime = ImageData[2 * EvenSpikeNum];  //存储上一个事件的时刻
				}
          
            }
            if(LearnRule == "Predict")
            {
                if(pattern == "SOM-STDP&RSTDP")
                    preLabel = WinnerDecide(Parameter.NeuronRelativeLabel, Parameter.EachLabelNeuronNum, SingleStreamResponse, false);
                else if(pattern == "SOM-RSTDP")
                    preLabel = WinnerDecide(Parameter.NeuronRelativeLabel, Parameter.EachLabelNeuronNum, SingleStreamResponse, true);

                Parameter.truelabel2predictlabel[InputClassLabel, preLabel] += 1;
                if (preLabel == InputClassLabel)
                {
                    Parameter.NumCorrect += 1;
                    Parameter.LabelCorrectNum[preLabel]++;
                }
            }
        }


        public static void SOMTrain(string pattern)
        {
            Stopwatch watch = new Stopwatch();
            /*--------------------som stage learning epoch ---------------------------------------/
            all-pair model  
                som-stdp&rstdp:  yale: 50
            nearest-pair model
                som-r-stdp:      yale: 50
            /--------------------som stage learning epoch---------------------------------------*/
            for (int externEp = 0; externEp < 50; externEp++) 
            {
                for (int Epoch = 0; Epoch < MAX_EPOCH; Epoch++)
                {
                    EP = Epoch;
                    watch.Start();
                    for (int i = 0; i < TrainNum; i++)
                    {
                        imgNum = Epoch * ImageOffest + i;
                        //Console.WriteLine(imgNum);
                        if ((imgNum) % 2000 == 0) 
                        {
                            DynamicThreasholdDecay(Parameter.DynamicThreshold_tao);
                        }
                        FeedforwardPass(TrainCodingSpike[imgNum], TrainLabel[imgNum], "SOM", imgNum, pattern);
                        
                    }
                    //训练结束，将权重写入文件(用于显示权重图像)
                    string WEIGHT_PATH = "../../../../DataSet/weight" + Epoch.ToString() + ".txt";
                    StreamWriter PureWeightRecord = new StreamWriter(WEIGHT_PATH);
                    for (int i = 0; i < Parameter.WeightTensor.GetLength(0); i++) // 
                        for (int j = 0; j < Parameter.WeightTensor.GetLength(1); j++) // 
                            PureWeightRecord.WriteLine(Parameter.WeightTensor[i, j]);
                    PureWeightRecord.Close();
                    VariableRefresh();
                    watch.Stop();
                    Console.WriteLine("{0}次小迭代训练结束，训练过程共消耗 {1} 分钟", Epoch.ToString(),(watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
                    watch.Reset();
                }
            }
            //动态阈值衰减
            DynamicThreasholdScaling(32); 
            //记录动态阈值
            string DYNAMIC_THRESHOLD_RECORD_PATH = "../../../../DataSet/DynamicThreshold.txt";
            StreamWriter DynamicThresholdRecord = new StreamWriter(DYNAMIC_THRESHOLD_RECORD_PATH);
            for (int Neuron = 0; Neuron < Parameter.OutputLayerNeurons; Neuron++)
                DynamicThresholdRecord.WriteLine(Parameter.DynamicThreshold[Neuron].ToString());
            DynamicThresholdRecord.Close();


            //标签
            watch.Start();
            nofire_imagenum = 0;
            Parameter.nofireimagelabel = new int[Parameter.NumClass];
            int[] EachLabelNum = new int[Parameter.NumClass];
            for(int imgNum = 0; imgNum < MNISTTrainNum; imgNum++)
            {
                    FeedforwardPass(TrainCodingSpike[imgNum], TrainLabel[imgNum], "ClassAssign", imgNum, pattern);
                    EachLabelNum[TrainLabel[imgNum]] += 1;

            }

            Console.WriteLine("标签过程没有发射脉冲的图像数为{0}", nofire_imagenum.ToString());
            for(int i = 0; i < Parameter.NumClass;i++)
            {
                Console.WriteLine("标签{0}没有发射脉冲的图像数为{1}",i.ToString(), Parameter.nofireimagelabel[i].ToString());
            }

            //记录每个输出神经元的累积响应
            string LABEL_NEURON_RECORD_PATH = "../../../../DataSet/label_neuron_max_all_pure.txt";
            StreamWriter LabelNeuronRecord = new StreamWriter(LABEL_NEURON_RECORD_PATH);
            for (int Neuron = 0; Neuron < Parameter.OutputLayerNeurons; Neuron++)
                for (int Label = 0; Label < Parameter.NumClass; Label++)
                    LabelNeuronRecord.WriteLine("神经元：{0}，标签{1}的累积响应为：{2}，该标签出现的次数为{3}", Neuron, Label, Parameter.LabelNeuronResponse[Neuron, Label], EachLabelNum[Label]);
            LabelNeuronRecord.Close();

            //打标签
            for (int i = 0; i < Parameter.LabelNeuronResponse.GetLength(0); i++)
                for (int j = 0; j < Parameter.LabelNeuronResponse.GetLength(1); j++)
                {
                    Parameter.LabelNeuronResponse[i, j] = (int)Math.Round(1.0 * Parameter.LabelNeuronResponse[i, j] / EachLabelNum[j] * (1 << Parameter.FIXED_POINT_4));
                    if (Parameter.LabelNeuronResponse[i, j] >= ((4096 << Parameter.FIXED_POINT_4) - 1))
                    {
                        Parameter.LabelNeuronResponse[i, j] = ((4096 << Parameter.FIXED_POINT_4) - 1);
                    }
                }
            Parameter.NeuronRelativeLabel = ClassAssign(Parameter.LabelNeuronResponse); // 打上标签


            //记录EachLabelNeuronNum
            string EachLabelNeuronNum_RECORD_PATH = "../../../../DataSet/EachLabelNeuronNum.txt";
            StreamWriter EachLabelNeuronNumRecord = new StreamWriter(EachLabelNeuronNum_RECORD_PATH);
            for (int Class = 0; Class < Parameter.NumClass; Class++)
                EachLabelNeuronNumRecord.WriteLine(Parameter.EachLabelNeuronNum[Class]);
            EachLabelNeuronNumRecord.Close();

            //记录标签
            string LABEL_RECORD_PATH = "../../../../DataSet/label_neuron_pure.txt";
            string LabelPlotPath = "../../../../DataSet/labelPlot.txt";
            StreamWriter LabelRecord = new StreamWriter(LABEL_RECORD_PATH);
            StreamWriter LabelPlotRecord = new StreamWriter(LabelPlotPath);
            for (int Neuron = 0; Neuron < Parameter.OutputLayerNeurons; Neuron++)
            {
                LabelPlotRecord.WriteLine(Parameter.NeuronRelativeLabel[Neuron]);
                LabelRecord.WriteLine("神经元{0}的标签为{1}", Neuron, Parameter.NeuronRelativeLabel[Neuron]);
            }
            for (int Class = 0; Class < Parameter.NumClass; Class++)
                LabelRecord.WriteLine("标签为{0}的神经元数量有{1}个", Class, Parameter.EachLabelNeuronNum[Class]);
            LabelRecord.Close();
            LabelPlotRecord.Close();

            // 打标签完毕
            watch.Stop();
            Console.WriteLine("标签标记过程结束，打标签共消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();
        }


        public static void RSTDP_Tuned(string pattern)
        {
            Stopwatch watch = new Stopwatch();

            if (pattern == "SOM-RSTDP")
                LabelInit(Parameter.NumClass);

            //for (int epoch = 0; epoch < RSTDP_EP; epoch++)
            for (int epoch = 0; epoch < RSTDP_EP; epoch++)
            {
                watch.Start();
                for(int img = 0; img < MNISTTrainNum; img++)
                {
                    if ((img + 1) % 1580 == 0)
                    {
                       DynamicThreasholdDecay(Parameter.DynamicThreshold_tao);
                    }
                    FeedforwardPass(TrainCodingSpike[img], TrainLabel[img], "R_STDP", img, pattern);
                }
                //训练结束，将权重写入文件
                string WEIGHT_PATH = "../../../../DataSet/RSTDPweight" + epoch.ToString() + ".txt";
                StreamWriter PureWeightRecord = new StreamWriter(WEIGHT_PATH);
                for (int i = 0; i < Parameter.WeightTensor.GetLength(0); i++) // 
                    for (int j = 0; j < Parameter.WeightTensor.GetLength(1); j++) // 
                        PureWeightRecord.WriteLine(Parameter.WeightTensor[i, j]);
                PureWeightRecord.Close();
                watch.Stop();
                Console.WriteLine("{0}次迭代训练结束，训练过程共消耗 {1} 分钟", epoch.ToString(),(watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
                watch.Reset();
            }
            //DynamicThreasholdScaling(74); //74
            //记录动态阈值
            string DYNAMIC_THRESHOLD_RECORD_PATH = "../../../../DataSet/strp_DynamicThreshold.txt";
            StreamWriter DynamicThresholdRecord = new StreamWriter(DYNAMIC_THRESHOLD_RECORD_PATH);
            for (int Neuron = 0; Neuron < Parameter.OutputLayerNeurons; Neuron++)
                DynamicThresholdRecord.WriteLine(Parameter.DynamicThreshold[Neuron].ToString());
            DynamicThresholdRecord.Close();
        }

        public static void Test(string pattern)
        {
            if (pattern == "SOM-RSTDP")
                LabelInit(Parameter.NumClass);
            Stopwatch watch = new Stopwatch();
            nofire_imagenum = 0;
            Parameter.truelabel2predictlabel = new int [Parameter.NumClass,Parameter.NumClass];
            Parameter.LabelCorrectNum = new int[Parameter.NumClass];
            watch.Start();
            Parameter.NumCorrect = 0;
            Parameter.EachLabelNum = new int[Parameter.NumClass];
            Parameter.category3_wrong_index = new ArrayList();

            string predictlabel_RECORD_PATH = "../../../../DataSet/predictlabel.txt";
            StreamWriter predictlabelRecord = new StreamWriter(predictlabel_RECORD_PATH);
            for (int TestImg = 0; TestImg < TestNum; TestImg++)
            {
                Parameter.image_for_category_fire_num = new int[Parameter.NumClass];
                imagefirenum = new int[Parameter.OutputLayerNeurons];
                FeedforwardPass(TestCodingSpike[TestImg], TestLabel[TestImg], "Predict", TestImg, pattern);
                predictlabelRecord.WriteLine(preLabel.ToString());
                Parameter.EachLabelNum[TestLabel[TestImg]]++;

                if (preLabel != TestLabel[TestImg] && TestLabel[TestImg] == 3)
                {
                    Parameter.category3_wrong_index.Add(TestImg);
                }
            }

            string category3_wrong_index_RECORD_PATH = "../../../../DataSet/category3_wrong_index.txt";
            StreamWriter category3_wrong_indexRecord = new StreamWriter(category3_wrong_index_RECORD_PATH);
            for (int i = 0; i < Parameter.category3_wrong_index.Count; i++)
            {
                category3_wrong_indexRecord.WriteLine(Parameter.category3_wrong_index[i].ToString());
            }
            category3_wrong_indexRecord.Close();


            double CorrectRate = ((double)Parameter.NumCorrect) / TestNum * 100;
            watch.Stop();
            Console.WriteLine("测试准确率为 {0}，测试过程共消耗 {1} 分钟", CorrectRate.ToString(".##"), (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();
            GlobalCorrectRate = CorrectRate;



            Console.WriteLine("测试过程没有发射脉冲的图像数为{0}", nofire_imagenum.ToString());
            predictlabelRecord.Close();

            for (int i = 0; i < Parameter.NumClass;i++)
            {
                Console.WriteLine("标签{0}的识别正确率为{1}", i.ToString(), ((double)Parameter.LabelCorrectNum[i] / Parameter.EachLabelNum[i]).ToString());
;           }


            //truelabel2predictlabel 
            string truelabel2predictlabel_RECORD_PATH = "../../../../DataSet/truelabel2predictlabel.txt";
            StreamWriter truelabel2predictlabelRecord = new StreamWriter(truelabel2predictlabel_RECORD_PATH);
            for (int truelabel = 0; truelabel < Parameter.NumClass; truelabel++)
            {
                for (int prelabel = 0; prelabel < Parameter.NumClass; prelabel++)
                {
                    truelabel2predictlabelRecord.WriteLine("真实标签{0}预测为{1}的个数为{2}", truelabel.ToString(), prelabel.ToString(), Parameter.truelabel2predictlabel[truelabel, prelabel].ToString());
                }
                truelabel2predictlabelRecord.WriteLine();
            }
            truelabel2predictlabelRecord.Close();
        }


        //(1)文件的读取
        public static void ReadEvenSpikeFromFile()
        {
            string line_data;
            int count = 0;
            while ((line_data = sr_train_label.ReadLine()) != null)  //训练标签
            {
                TrainLabel[count++] = Convert.ToInt32(line_data);
            }
            count = 0;
            while ((line_data = sr_test_label.ReadLine()) != null)  //测试标签
            {
                TestLabel[count++] = Convert.ToInt32(line_data);
            }
            count = 0;
            while((line_data = sr_train_data.ReadLine()) != null)
            {
                string[] str_data = line_data.Split(' ');

                TrainCodingSpike[count] = new int[str_data.Length - 1];

                for (int i = 0; i < str_data.Length - 1; i++)
                {
                    TrainCodingSpike[count][i] = Convert.ToInt32(str_data[i]);
                }
                count++;
            }
            count = 0;
            while((line_data = sr_test_data.ReadLine()) != null)
            {
                string[] str_data = line_data.Split(' ');
                TestCodingSpike[count] = new int[str_data.Length - 1];

                for (int i = 0; i < str_data.Length - 1; i++)
                {
                    TestCodingSpike[count][i] = Convert.ToInt32(str_data[i]);
                }
                count++;
            }
        }

        //(2)--初始化权重
        public static void WeightRandomInit()
        {
            StreamWriter InitWeightRecord = new StreamWriter(InitWeightPath);
            for (int i = 0; i < Parameter.WeightTensor.GetLength(0);i++)
            {
                for(int j = 0; j < Parameter.WeightTensor.GetLength(1);j++)
                {
                    Parameter.WeightTensor[i, j] = (int)Math.Round(WeightRandomHandle.NextDouble() * WeightFactor * (1 << Parameter.FIXED_POINT_12));
                    InitWeightRecord.WriteLine(Convert.ToString(Parameter.WeightTensor[i, j],16));
                }
            }
            InitWeightRecord.Close();
        }
        //(3)脉冲数据读取和权重初始化
        public static void PriorProcess()
        {
            Stopwatch watch = new Stopwatch();
            watch.Start();
            WeightRandomInit();
            ReadEvenSpikeFromFile();
            ExpLookupTable();
            watch.Stop();
            Console.WriteLine("读取文件结束，读取文件共消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();
        }

        //(4)查表
        public static void ExpLookupTable()
        {
            StreamWriter PreTrace_EXP_LUTRecord = new StreamWriter(PreTrace_EXP_LUT_Path);
            StreamWriter PostTrace_EXP_LUTRecord = new StreamWriter(PostTrace_EXP_LUT_Path);
            StreamWriter Vm_EXP_LUTRecord = new StreamWriter(Vm_EXP_LUT_Path);
            for(int t = 0; t < Parameter.SampleTimeWindow; t++)
            {
                Parameter.PreExpTable[t] = (int)Math.Round(Math.Exp(-t / Parameter.PreSpike_tao) * (1 << Parameter.FIXED_POINT_4));
                Parameter.PostExpTable[t] = (int)Math.Round(Math.Exp(-t / Parameter.PostSpike_tao) * (1 << Parameter.FIXED_POINT_4));
                Parameter.VmExpTable[t] = (int)Math.Round(Math.Exp(-t / Parameter.V_tao) * (1 << Parameter.FIXED_POINT_8));

                PreTrace_EXP_LUTRecord.WriteLine(Convert.ToString(Parameter.PreExpTable[t],16));
                PostTrace_EXP_LUTRecord.WriteLine(Convert.ToString(Parameter.PostExpTable[t],16));
                Vm_EXP_LUTRecord.WriteLine(Convert.ToString(Parameter.VmExpTable[t],16));
            }
            PreTrace_EXP_LUTRecord.Close();
            PostTrace_EXP_LUTRecord.Close();
            Vm_EXP_LUTRecord.Close();
        }


        //5)标签函数
        public static int[] ClassAssign(int[,] OutputResponseInfo) // 输入100*10的矩阵，即每一个输出神经元对于每一类的响应
        {
            int[] LabelNeuron = new int[Parameter.OutputLayerNeurons];

            for (int Neuron = 0; Neuron < OutputResponseInfo.GetLength(0); Neuron++) // 将每个输出神经元对于每一类的总响应先排序，再将响应最大的一类作为标签赋给该神经元
            {
                int[] LabelAddr = new int[Parameter.OutputLayerNeurons];
                for (int Addr = 0; Addr < LabelAddr.Length; Addr++)
                    LabelAddr[Addr] = Addr;
                for (int ii = 0; ii < OutputResponseInfo.GetLength(1); ii++)// 排序
                {

                    if (double.IsNaN(OutputResponseInfo[Neuron, ii]))
                    {
                        OutputResponseInfo[Neuron, ii] = 0;
                    }
                    for (int jj = ii; jj < OutputResponseInfo.GetLength(1); jj++)
                    {
                        if (double.IsNaN(OutputResponseInfo[Neuron, jj]))
                        {
                            OutputResponseInfo[Neuron, jj] = 0;
                        }
                        if (OutputResponseInfo[Neuron, ii] < OutputResponseInfo[Neuron, jj])
                        {
                            int temp = OutputResponseInfo[Neuron, ii];
                            OutputResponseInfo[Neuron, ii] = OutputResponseInfo[Neuron, jj]; //exchange 
                            OutputResponseInfo[Neuron, jj] = temp;

                            int temp_addr = LabelAddr[ii];
                            LabelAddr[ii] = LabelAddr[jj];
                            LabelAddr[jj] = temp_addr;
                        }
                    }
                }
                LabelNeuron[Neuron] = LabelAddr[0];
                Parameter.EachLabelNeuronNum[LabelAddr[0]] += 1;
            }
            return LabelNeuron;
        }

        //6)决策函数
        public static int WinnerDecide(int[] LabelNeuron, int[] EachLabelNeuronNum, int[] OutputResponseInfo, bool R_STDP_Flag)
        {

            double[] EachLabelResponse = new double[Parameter.NumClass];
            for (int TotalNeuron = 0; TotalNeuron < LabelNeuron.Length; TotalNeuron++)
            {
                int n = LabelNeuron[TotalNeuron];
                EachLabelResponse[n] += OutputResponseInfo[TotalNeuron]; // 累计每个标签的神经元响应
            }
                
            for (int i = 0; i < EachLabelResponse.Length; i++) // 响应归一化
            {
                if (R_STDP_Flag)
                {
                    if (Parameter.NumClass == 10)
                    {
                        if(i == 9)
                            EachLabelResponse[i] = EachLabelResponse[i]/22;
                        else
                            EachLabelResponse[i] = EachLabelResponse[i] / 26;
                    }
                    else
                        EachLabelResponse[i] = EachLabelResponse[i];
                }
                    
                else
                {
                    EachLabelResponse[i] = EachLabelResponse[i] / EachLabelNeuronNum[i];
                    
                    if (double.IsNaN(EachLabelResponse[i]))
                    {
                        EachLabelResponse[i] = 0;
                    }
                }
            }

           
            int WinnerIndex = 0;
            double MaxLabelResponse = EachLabelResponse[0];
            for (int i = 1; i < Parameter.NumClass; i++)
            {
                if (MaxLabelResponse < EachLabelResponse[i])
                {
                    WinnerIndex = i;
                    MaxLabelResponse = EachLabelResponse[i];
                }
            }
            return WinnerIndex;
        }

        //7)相关全局变量清零（输出神经元响应、标签，每一类标签的样本数，预测正确的样本个数）
        public static void VariableRefresh()
        {
            Parameter.LabelNeuronResponse = new int[Parameter.OutputLayerNeurons, Parameter.NumClass]; // 记录每个输出神经元对于标签的响应情况
            Parameter.NeuronRelativeLabel = new int[Parameter.OutputLayerNeurons]; // 记录每个输出神经元对应的标签
            Parameter.EachLabelNeuronNum = new int[Parameter.NumClass];
            Parameter.NumCorrect = 0;
        }

        //8)动态阈值缩放
        public static void DynamicThreasholdScaling(int div)
        {
            for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
            {
                Parameter.DynamicThreshold[i] = ((div) * Parameter.DynamicThreshold[i]) >> (Parameter.FIXED_POINT_8);
            }      
        }

        //9)动态阈值衰减
        public static void DynamicThreasholdDecay(double tao)
        {
            for (int i = 0; i < Parameter.OutputLayerNeurons; i++)
                Parameter.DynamicThreshold[i] = (Parameter.DynamicThreshold[i] * ((1 << 8) - (int)Math.Round((1.0 / tao) * (1 << 8)))) >> 8;
        }


        //10)读取SOM阶段得到的标签和权重
        public static void ReadLabelAndWeightFromSOM()
        {

            FileStream fs_label_data = new FileStream(LabelPath, FileMode.Open);
            StreamReader sr_label_data = new StreamReader(fs_label_data);
            FileStream fs_weight_data = new FileStream(WeightPath, FileMode.Open);
            StreamReader sr_weight_data = new StreamReader(fs_weight_data);
            FileStream fs_dynamic_data = new FileStream(DynamicPath, FileMode.Open);
            StreamReader sr_dynamic_data = new StreamReader(fs_dynamic_data);

            FileStream fs_EachLabelNeuronNum_data = new FileStream(EachLabelNeuronNumPath, FileMode.Open);
            StreamReader sr_EachLabelNeuronNum_data = new StreamReader(fs_EachLabelNeuronNum_data);


            Stopwatch watch = new Stopwatch();
            watch.Start();
            string line_data;
            int count = 0;
            while ((line_data = sr_label_data.ReadLine()) != null)  //SOM label
            {
                Parameter.NeuronRelativeLabel[count++] = Convert.ToInt32(line_data);
            }
            
            
            count = 0;
            while ((line_data = sr_weight_data.ReadLine()) != null)  //weight
            {
                Parameter.WeightTensor[count / 784,count % 784] = Convert.ToInt32(line_data);
                count++;
            }
            count = 0;
            while ((line_data = sr_dynamic_data.ReadLine()) != null)  //weight
            {
                Parameter.DynamicThreshold[count++] = Convert.ToInt32(line_data);
            }
           
            count = 0;
            while ((line_data = sr_EachLabelNeuronNum_data.ReadLine()) != null)  //weight
            {
                Parameter.EachLabelNeuronNum[count++] = Convert.ToInt32(line_data);
            }
            watch.Stop();
            Console.WriteLine("读取文件结束，读取文件共消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();
        }


        public static void ReadLabelAndWeightFromRSTDP()
        {

            FileStream fs_weight_data = new FileStream(RSTDP_WEIGHT_PATH, FileMode.Open);
            StreamReader sr_weight_data = new StreamReader(fs_weight_data);
            FileStream fs_dynamic_data = new FileStream(RSTDP_DynamicPath, FileMode.Open);
            StreamReader sr_dynamic_data = new StreamReader(fs_dynamic_data);
            Stopwatch watch = new Stopwatch();
            watch.Start();
            string line_data;
            int count = 0;
            while ((line_data = sr_weight_data.ReadLine()) != null)  //weight
            {
                Parameter.WeightTensor[count / Parameter.InputLayerNeurons, count % Parameter.InputLayerNeurons] = Convert.ToInt32(line_data);
                count++;
            }
            count = 0;
            while ((line_data = sr_dynamic_data.ReadLine()) != null)  //weight
            {
                Parameter.DynamicThreshold[count++] = Convert.ToInt32(line_data);
            }
            watch.Stop();
            Console.WriteLine("读取文件结束，读取文件共消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();

            sr_weight_data.Close();
            sr_dynamic_data.Close();
        }


        public static void LabelInit(int classnum)
        {
            StreamWriter InitLabekRecord = new StreamWriter(LabelPath);
            for (int i = 0; i < 256; i++)
            {
                if(classnum == 10)
                    Parameter.NeuronRelativeLabel[i] = i / 26;
                else if(classnum == 8)
                    Parameter.NeuronRelativeLabel[i] = i / 32;
                else if(classnum == 4)
                    Parameter.NeuronRelativeLabel[i] = i / 64;
                else if(classnum == 3)
                    Parameter.NeuronRelativeLabel[i] = i / 86;
                //Console.WriteLine(Parameter.NeuronRelativeLabel[i]);
                InitLabekRecord.WriteLine(Parameter.NeuronRelativeLabel[i].ToString());
            }
            InitLabekRecord.Close();
        }
    }
}