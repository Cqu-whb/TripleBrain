﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;
using System.Linq;
using System.IO;

namespace CODING
{

    public static class PoisionCoding
    {

        public const string DataPath = "../../../../DataSet/";

       /*
        //MNIST
        private const int WriterFileTrainNum = 60000;
        private const int WriterFileTestNum = 10000;
        public const string MNISTTrainDataName = "mnist_train.csv";
        public const string MNISTTestDataName = "mnist_test.csv";
        public const string MNISTTrainCodingName = "mnist_train_coding.txt";
        public const string MNISTTestCodingName = "mnist_test_coding.txt";
        private const string MNISTTrainLabelName = "mnist_train_label.txt";
        private const string MNISTTestLabelName = "mnist_test_label.txt";
        private const int MNISTTrainDataNum = 60000;
        private const int MNISTTestDataNum = 10000;
        private const int MNISTColumnSize = 28;
        private const int MNISTRowSize = 28;
       */
        

        /*
        //orl
        private const int WriterFileTrainNum = 60;
        private const int WriterFileTestNum = 40;
        public const string MNISTTrainDataName = "orl_face_train.csv";
        public const string MNISTTestDataName = "orl_face_test.csv";
        public const string MNISTTrainCodingName = "orl_face_train_coding.txt";
        public const string MNISTTestCodingName = "orl_face_test_coding.txt";
        private const string MNISTTrainLabelName = "orl_face_train_label.txt";
        private const string MNISTTestLabelName = "orl_face_test_label.txt";
        private const int MNISTTrainDataNum = 60;
        private const int MNISTTestDataNum = 40;
        private const int MNISTColumnSize = 32;
        private const int MNISTRowSize = 32;
        */
        

        /*
        //yale
        private const int WriterFileTrainNum = 382;
        private const int WriterFileTestNum = 257;
        public const string MNISTTrainDataName = "yale_face_train.csv";
        public const string MNISTTestDataName = "yale_face_test.csv";
        public const string MNISTTrainCodingName = "yale_face_train_coding.txt";
        public const string MNISTTestCodingName = "yale_face_test_coding.txt";
        private const string MNISTTrainLabelName = "yale_face_train_label.txt";
        private const string MNISTTestLabelName = "yale_face_test_label.txt";
        private const int MNISTTrainDataNum = 382;
        private const int MNISTTestDataNum = 257;
        private const int MNISTColumnSize = 32;
        private const int MNISTRowSize = 32;
        */
        

        //MNIST 用于modelsim仿真
        public const string MNISTTrainEventName = "mnist_train_event.txt";
        public const string MNISTTestEventName = "mnist_test_event.txt";

        /*
        //Fashion-MNIST
        private const int WriterFileTrainNum = 60000;
        private const int WriterFileTestNum = 10000;
        public const string MNISTTrainDataName = "fashion_mnist_train.csv";
        public const string MNISTTestDataName = "fashion_mnist_test.csv";
        public const string MNISTTrainCodingName = "fashion_mnist_train_coding.txt";
        public const string MNISTTestCodingName = "fashion_mnist_test_coding.txt";
        private const string MNISTTrainLabelName = "fashion_mnist_train_label.txt";
        private const string MNISTTestLabelName = "fashion_mnist_test_label.txt";
        private const int MNISTTrainDataNum = 60000;
        private const int MNISTTestDataNum = 10000;
        private const int MNISTColumnSize = 28;
        private const int MNISTRowSize = 28;
        */
         
        
        //ETH-80
        private const int WriterFileTrainNum = 2296;
        private const int WriterFileTestNum = 984;
        public const string MNISTTrainDataName = "eth80_train.csv";
        public const string MNISTTestDataName = "eth80_test.csv";
        public const string MNISTTrainCodingName = "eth80_train_coding.txt";
        public const string MNISTTestCodingName = "eth80_test_coding.txt";
        private const string MNISTTrainLabelName = "eth80_train_label.txt";
        private const string MNISTTestLabelName = "eth80_test_label.txt";
        private const int MNISTTrainDataNum = 2296;
        private const int MNISTTestDataNum = 984;
        private const int MNISTColumnSize = 32;
        private const int MNISTRowSize = 32;
        
        


        private const int SampleTimeWindow = 200;
        private const int TotalTime = 1000;
        private const int TimeUnit = 1;  //时间单位
        static Random rnd = new Random(0);

        public static void Coding(string CodingType)
        {
            string TrainCodingSpike_Path = DataPath + MNISTTrainCodingName;
            string TestCodingSpike_Path = DataPath + MNISTTestCodingName;
            string TrainLabel_Path = DataPath + MNISTTrainLabelName;
            string TestLabel_Path = DataPath + MNISTTestLabelName;

            StreamWriter TrainCodingSpikeRecord = new StreamWriter(TrainCodingSpike_Path);
            StreamWriter TestCodingSpikeRecoed = new StreamWriter(TestCodingSpike_Path);
            StreamWriter TrainLabelRecord = new StreamWriter(TrainLabel_Path);
            StreamWriter TestLabelRecord = new StreamWriter(TestLabel_Path);


            //用于modelsim仿真
            string TrainEventPackage_Path = DataPath + MNISTTrainEventName;
            string TestEventPackag_Path = DataPath + MNISTTestEventName;
            StreamWriter TrainEventPackagRecord = new StreamWriter(TrainEventPackage_Path);
            StreamWriter TestEventPackagRecoed = new StreamWriter(TestEventPackag_Path);





            int[][] MNISTTrainData = new int[MNISTTrainDataNum][];
            for (int i = 0; i < MNISTTrainDataNum; i++)
                MNISTTrainData[i] = new int[MNISTColumnSize * MNISTRowSize];
            int[] MNISTTrainLabel = new int[MNISTTrainDataNum];

            int[][] MNISTTestData = new int[MNISTTestDataNum][];
            for (int i = 0; i < MNISTTestDataNum; i++)
                MNISTTestData[i] = new int[MNISTColumnSize * MNISTRowSize];
            int[] MNISTTestLabel = new int[MNISTTestDataNum];

            CSVDataRead(DataPath, MNISTTrainDataName, ref MNISTTrainData, ref MNISTTrainLabel);
            CSVDataRead(DataPath, MNISTTestDataName, ref MNISTTestData, ref MNISTTestLabel);

            //编码
            for(int TrainNum = 0; TrainNum < WriterFileTrainNum; TrainNum++)
            {
                TrainLabelRecord.WriteLine(MNISTTrainLabel[TrainNum]);
                bool[][] TrainPoisongSpk = new bool[SampleTimeWindow][];

                if (CodingType == "PoisionCoding")
                    TrainPoisongSpk = Image2PoisionSpikeTrain(MNISTTrainData[TrainNum]);
                else if (CodingType == "MultiSpikeCoing")
                    TrainPoisongSpk = Image2MultiSpikeCodingSpikeTrain(MNISTTrainData[TrainNum]);
                else if (CodingType == "TemporalCoding")
                    TrainPoisongSpk = Image2TemporalCodingSpikeTrain(MNISTTrainData[TrainNum]);

                for (int t = 0; t < SampleTimeWindow; t++)
                {
                    for(int PixNum = 0; PixNum < MNISTColumnSize * MNISTRowSize; PixNum++)
                    {
                        if(TrainPoisongSpk[t][PixNum])
                        {
                            TrainCodingSpikeRecord.Write(t.ToString() + ' ' + PixNum.ToString() + ' ');
                            int event_package = (PixNum << 8) + t;
                            TrainEventPackagRecord.Write(Convert.ToString(event_package,16) + ' ');
                        }
                    }
                    //TrainCodingSpikeRecord.Write('\n');
                }
                TrainCodingSpikeRecord.Write('\n');
                TrainEventPackagRecord.Write('\n');
            }
            TrainLabelRecord.Close();
            TrainCodingSpikeRecord.Close();


            for(int TestNum = 0; TestNum < WriterFileTestNum; TestNum++)
            {
                TestLabelRecord.WriteLine(MNISTTestLabel[TestNum]);

                bool[][] TestPoisionSpk = new bool[SampleTimeWindow][];
                if (CodingType == "PoisionCoding")
                    TestPoisionSpk = Image2PoisionSpikeTrain(MNISTTestData[TestNum]);
                else if (CodingType == "MultiSpikeCoing")
                    TestPoisionSpk = Image2MultiSpikeCodingSpikeTrain(MNISTTestData[TestNum]);
                else if (CodingType == "TemporalCoding")
                    TestPoisionSpk = Image2TemporalCodingSpikeTrain(MNISTTestData[TestNum]);

                for (int t = 0; t < SampleTimeWindow; t++)
                {
                    for(int PixNum = 0; PixNum < MNISTColumnSize * MNISTRowSize;PixNum++)
                    {
                        if(TestPoisionSpk[t][PixNum])
                        {
                            //TestCodingSpikeRecoed.Write(PixNum.ToString() + ',');
                            TestCodingSpikeRecoed.Write(t.ToString() + ' ' + PixNum.ToString() + ' ');
                            int event_package = (PixNum << 8) + t;
                            TestEventPackagRecoed.Write(Convert.ToString(event_package, 16) + ' ');
                        }
                    }
                   // TestCodingSpikeRecoed.Write('\n');   
                }
                TestCodingSpikeRecoed.Write('\n');
                TestEventPackagRecoed.Write('\n');
            }
            TestLabelRecord.Close();
            TestCodingSpikeRecoed.Close();

            TrainEventPackagRecord.Close();
            TestEventPackagRecoed.Close();
        }



        //读取CSV文件数据
        public static void CSVDataRead(string Filepath, string DataName, ref int[][] Data, ref int[] Label)
        {
            using (StreamReader FileData = new StreamReader(Filepath + DataName))
            {
                string Handle;
                string[] Temp;
                int ii = 0;
                while ((Handle = FileData.ReadLine()) != null)
                {
                    Temp = Handle.ToString().Split(',');
                    int jj = 0;
                    double Tempv = 0;
                    foreach (string j in Temp)
                    {
                        if (jj == 0)
                        {
                            Tempv = Convert.ToDouble(j);
                            Label[ii] = Convert.ToInt32(Tempv);  //标签
                        }
                        else
                        {
                            Data[ii][jj - 1] = Convert.ToInt32(j);  //数据
                        }
                        jj += 1;
                    }
                    ii += 1;
                }
            }
        }


        public static bool[][] Image2PoisionSpikeTrain(int[] ImageData)  //FrequencyOffest 触发脉冲频率的偏置
        {
            bool[][] SpikeTrain = new bool[SampleTimeWindow][];
            for (int Time = 0; Time < SampleTimeWindow; Time++)
                SpikeTrain[Time] = new bool[ImageData.GetLength(0)];  //为每一行指定行中的元素个数，且元素值为bool型
            for (int Address = 0; Address < ImageData.Length; Address++)
            {
                if (ImageData[Address] > 0) //像素值大于0
                {
                    int TimeIndex = 0;
                    double Frequency;
                    Frequency = ImageData[Address] / 4.0;
                    double Internal = TotalTime / Frequency;
                    while (TimeIndex < SampleTimeWindow)
                    {
                        int TimeInternal;
                        TimeInternal = poisson(Internal, TimeUnit);
                        if (TimeInternal != 0)
                            TimeIndex += TimeInternal;
                        else
                            TimeIndex += 1;
                        if (TimeIndex < SampleTimeWindow)
                            SpikeTrain[TimeIndex][Address] = true;
                    }
                }
            }
            return SpikeTrain;
        }


        public static double ngtIndex(double lam)
        {
            double dec = rnd.NextDouble();
            while (dec == 0)
                dec = rnd.NextDouble();
            return -Math.Log(dec) / lam;
        }

        public static int poisson(double lam, double time)
        {
            int count = 0;
            while (true)
            {
                time -= ngtIndex(lam);
                if (time > 0)
                    count++;
                else
                    break;
            }
            return count;
        }

        public static bool [][] Image2TemporalCodingSpikeTrain(int[] ImageData)
        {
            bool[][] SpikeTrain = new bool[SampleTimeWindow][];
            for (int Time = 0; Time < SampleTimeWindow; Time++)
                SpikeTrain[Time] = new bool[ImageData.GetLength(0)];  //为每一行指定行中的元素个数，且元素值为bool型

            for (int Address = 0; Address < ImageData.Length; Address++)
            {
                if (ImageData[Address] > 0) //像素值大于0
                {
                    SpikeTrain[(int)(SampleTimeWindow*(1 - ImageData[Address] / 256.0))][Address] = true;
                }
            }
            return SpikeTrain;
        }


        public static bool [][] Image2MultiSpikeCodingSpikeTrain(int[] ImageData)
        {

            bool[][] SpikeTrain = new bool[SampleTimeWindow][];
            for (int Time = 0; Time < SampleTimeWindow; Time++)
                SpikeTrain[Time] = new bool[ImageData.GetLength(0)];  //为每一行指定行中的元素个数，且元素值为bool型
            for (int Address = 0; Address < ImageData.Length; Address++)
            {
                if (ImageData[Address] > 80) //像素值大于0
                {
                    double vth = 0.0;
                    int prefiretime = 0;
                    for (int t = 0; t < SampleTimeWindow; t++)
                    {
                        int diet = t - prefiretime;
                        vth = 0.0004 * ImageData[Address] * diet;
                        if (vth >= 1.0)
                        {
                            vth = 0.0;
                            prefiretime = t;
                            SpikeTrain[t][Address] = true;
                        }
                    }
                }
            }
            return SpikeTrain;
        }
    }
}