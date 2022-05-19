using System;
using MathNet.Numerics.LinearAlgebra;  //安装方法：工具-->NuGet包管理器-->程序包管理器控制台-->在下方命令行界面输入：Install-Package MathNet.Numerics -Version 4.7.0
using MathNet.Numerics.Data.Matlab;    //安装方法：工具-->NuGet包管理器-->程序包管理器控制台-->在下方命令行界面输入：Install-Package MathNet.Numerics.Data.Matlab -Version 3.2.0
using System.Diagnostics;
using System.IO;
using System.Collections;
namespace Read_mat
{
    class Program
    {
        /*
        //posture_dvs
        public static string Path = @"..\..\..\..\..\..\Common_Datasets\DVS\POSTURE_DVS\Comp_Linear\";
        public static string train_event_name = "train_coding.mat";
        public static string test_event_name = "test_coding.mat";
        public static string train_label_name = "train_labels.mat";
        public static string test_label_name = "test_labels.mat";
        public static int TimeWindow = 200;  //超出该时间窗口的事件抛弃
        */

        /*
        //card dvs
        public static string Path = @"..\..\..\..\..\..\Common_Datasets\DVS\CARD_DVS\Comp_Linear\";
        public static string train_event_name = "train_coding.mat";
        public static string test_event_name = "test_coding.mat";
        public static string train_label_name = "train_labels.mat";
        public static string test_label_name = "test_labels.mat";
        public static int TimeWindow = 200;  //超出该时间窗口的事件抛弃
        */

        /*
        //mnist-dvs
        public static string Path = @"..\..\..\..\..\..\Common_Datasets\DVS\MNIST_DVS\Comp_Linear\";
        public static string train_event_name = "train_coding.mat";
        public static string test_event_name = "test_coding.mat";
        public static string train_label_name = "train_labels.mat";
        public static string test_label_name = "test_labels.mat";
        public static int TimeWindow = 200;  //超出该时间窗口的事件抛弃
        */
        

        
        //n-mnist
        public static string Path = @"..\..\..\..\..\..\Common_Datasets\DVS\N_MNIST\Comp_Linear\";
        public static string train_event_name = "train_coding.mat";
        public static string test_event_name = "test_coding.mat";
        public static string train_label_name = "train_labels.mat";
        public static string test_label_name = "test_labels.mat";
        public static int TimeWindow = 200;  //超出该时间窗口的事件抛弃
        

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            //Matrix<double> mat_matrix = MatlabReader.Read<double>(@"..\..\..\..\..\..\Common_Datasets\Frame_based\MNIST\Original_Denoising_mat\train_images_60k.mat");
            Matrix<double> train_event_matrix = MatlabReader.Read<double>(Path + train_event_name);
            Matrix<double> test_event_matrix = MatlabReader.Read<double>(Path + test_event_name);
            Matrix<double> train_label_matrix = MatlabReader.Read<double>(Path + train_label_name);
            Matrix<double> test_label_matrix = MatlabReader.Read<double>(Path + test_label_name);

            //string TRAIN_LABEL_PATH = Path + "posture_train_label.txt";
            //string TRAIN_LABEL_PATH = Path + "cards_train_label.txt";
            //string TRAIN_LABEL_PATH = Path + "mnistdvs_train_label.txt";
            string TRAIN_LABEL_PATH = Path + "nmnist_train_label.txt";
            WriteLabel2File(train_label_matrix, TRAIN_LABEL_PATH);


            //string TEST_LABEL_PATH = Path + "posture_test_label.txt";
            //string TEST_LABEL_PATH = Path + "cards_test_label.txt";
            //string TEST_LABEL_PATH = Path + "mnistdvs_test_label.txt";
            string TEST_LABEL_PATH = Path + "nmnist_test_label.txt";
            WriteLabel2File(test_label_matrix, TEST_LABEL_PATH);


            //string TRAIN_EVENT_PATH = Path + "posture_train_coding.txt";
            //string TRAIN_EVENT_PATH = Path + "cards_train_coding.txt";
            //string TRAIN_EVENT_PATH = Path + "mnistdvs_train_coding.txt";
            string TRAIN_EVENT_PATH = Path + "nmnist_train_coding.txt";
            WriteEvent2File(train_event_matrix, TRAIN_EVENT_PATH);


            //string TEST_EVENT_PATH = Path + "posture_test_coding.txt";
            //string TEST_EVENT_PATH = Path + "cards_test_coding.txt";
            //string TEST_EVENT_PATH = Path + "mnistdvs_test_coding.txt";
            string TEST_EVENT_PATH = Path + "nmnist_test_coding.txt";
            WriteEvent2File(test_event_matrix, TEST_EVENT_PATH);


            //for (int i = 0; i < 784; i++)  //打印一张图像的像素，检查是否正确
            //    Console.WriteLine("pixel[{0}] = {1}", i, mat_matrix[0, i]);
        }



        static void WriteLabel2File(Matrix<double> mat_matrix,string label_path)
        {
            Stopwatch watch = new Stopwatch();
            //string LABEL_PATH = Path + "posture_train_label.txt";
            StreamWriter TrainLabelRecord = new StreamWriter(label_path);
            for (int i = 0; i < mat_matrix.RowCount; i++)
                TrainLabelRecord.WriteLine(mat_matrix[i,0]);
            TrainLabelRecord.Close();
            watch.Stop();
            Console.WriteLine("标签写入结束，过程共消耗 {0} 分钟",(watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();
        }


        static void WriteEvent2File(Matrix<double> mat_matrix, string event_path)
        {
            Stopwatch watch = new Stopwatch();
            ArrayList EachSampleEvent = new ArrayList();
            StreamWriter EventRecord = new StreamWriter(event_path);
            int tmax = 0;
            for (int i = 0; i < mat_matrix.ColumnCount; i++)
            {

                if (mat_matrix[0, i] == -1) //图像结束标志位
                {
                    for (int j = 0; j < EachSampleEvent.Count; j++)
                    {
                        if (j % 2 == 1)
                        {
                            if (TimeWindow > Convert.ToInt32(EachSampleEvent[j]))  // if aer timesatmp > TimeWindow   throw away!
                            {
                                EventRecord.Write(EachSampleEvent[j]); //先写时间
                                EventRecord.Write(' ');
                                EventRecord.Write(Convert.ToInt32(EachSampleEvent[j - 1]) - 1); //再写地址(addr-1由于matlab地址是从1开始)
                                if (Convert.ToInt32(EachSampleEvent[j - 1]) - 1 >= 1024)
                                {
                                    Console.WriteLine("overflow!");
                                    //Console.ReadKey();
                                }
                                EventRecord.Write(' ');
                                if (tmax <= Convert.ToInt32(EachSampleEvent[j]))
                                {
                                    tmax = Convert.ToInt32(EachSampleEvent[j]);
                                }
                                //if (Convert.ToInt32(EachSampleEvent[j - 1]) >= 1024)
                                //{
                                //    Console.WriteLine(EachSampleEvent[j - 1]);
                                //    Console.WriteLine("overflow!!");
                                //}
                            }
                        }
                    }
                    EventRecord.WriteLine();
                    EachSampleEvent = new ArrayList();
                    continue; //跳出当前循环
                }
                EachSampleEvent.Add(mat_matrix[0, i]);
            }

            Console.WriteLine("tmax = {0}", tmax);
            EventRecord.Close();
            watch.Stop();
            Console.WriteLine("脉冲写入结束，过程共消耗 {0} 分钟", (watch.ElapsedMilliseconds / 1000 / 60.0).ToString("F3"));
            watch.Reset();

        }
    }
}
