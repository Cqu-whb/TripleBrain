using System;
using System.Collections.Generic;
using System.Text;
using System.Collections;
using System.Linq;
using System.IO;
using CODING;
using SOM_RSTDPModel;
using PARAMETER;

namespace SOM_RSTDP
{

    class Program
    {

        public static string [] CodingTheme = new string[] { "PoisionCoding", "MultiSpikeCoing", "TemporalCoding" };
        public static string[] Pattern = new string[] { "SOM-STDP&RSTDP", "SOM-RSTDP" };
        public static string ResultPath = "../../../../Result/all_pair SOM-STDP&R-STDP.txt";
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            string pattern = Pattern[0];
            //bool a = false;
            //int x = 125;
            //x = x & (Convert.ToInt32(!a));
            //int b = Convert.ToInt32(!a);
            //Console.WriteLine(b);
            //Console.WriteLine(x);

            /***************速率编码**********************/
            PoisionCoding.Coding(CodingTheme[0]);
            Console.WriteLine("Coding finished!");
            ///************脉冲数据的读取和权重的初始化*********/
            Som_rstdpModel.PriorProcess();
            /************SOM训练和测试****************************/

            if (pattern == "SOM-STDP&RSTDP")
            {
                Som_rstdpModel.SOMTrain(pattern);
                //Som_rstdpModel.ReadLabelAndWeightFromSOM();
                Som_rstdpModel.Test(pattern);
            }


            ///************RSTDP微调和测试****************************/
            //// Som_rstdpModel.ReadLabelAndWeightFromSOM();
            Som_rstdpModel.RSTDP_Tuned(pattern);
            double maxcorrectrate = 0.0;
            int fac = 0;
            //for (int factor = 50; factor <= 120; factor++)
            for (int factor = 20; factor <= 120; factor++) 
            {
                Console.WriteLine("scaling系数为{0}下的识别率为:", factor.ToString());
                Som_rstdpModel.ReadLabelAndWeightFromRSTDP();
                Som_rstdpModel.DynamicThreasholdScaling(factor); //74
                Som_rstdpModel.Test(pattern);
                Console.WriteLine();
                if (Som_rstdpModel.GlobalCorrectRate > maxcorrectrate)
                {
                    maxcorrectrate = Som_rstdpModel.GlobalCorrectRate;
                    fac = factor;
                }
            }
            Console.WriteLine("最大的识别率为{0},对应的scaling系数为{1}", maxcorrectrate.ToString(".##"), fac.ToString());
            // save accuracy result
            StreamWriter ResultRecord = new StreamWriter(ResultPath);
            ResultRecord.WriteLine("accuracy:" + maxcorrectrate.ToString(".##") + "%");
            ResultRecord.Close();
        }
    }
}
