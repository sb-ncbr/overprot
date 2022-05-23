using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Net.Http;
using System.Linq;
using Cif;
using Cif.Filtering;
using Cif.Tables;

namespace StructureCutter
{
    public static class Debugging
    {   
        enum AlignMethod { None, Align, Super, Cealign };

        public static bool ArgumentParserTest(string[] args){
            ArgumentParser parser = new ArgumentParser("StructureCutter 0.9");
            var optDomainListFile = parser.AddStringArg("DOMAIN_LIST_FILE")
                .AddHelp("JSON file with the list of domains to be downloaded, in format")
                .AddHelp("[{\"pdb\": pdb, \"domain\": domain_name, \"chain_id\": chain_id, \"ranges\": ranges}*]")
                .AddHelp("or")
                .AddHelp("[[pdb, domain_name, chain_id, ranges]*]");

            Argument<string> os = parser.AddStringArg("-s");
            os.AddAction(() => Console.WriteLine(os.Value));
            Argument<int> oi = parser.AddIntArg("--count-n", "-i").SetChoices(1, 2, 3, 4, 5).SetDefault(5);
            oi.AddAction(() => Console.WriteLine(oi.Value));
            Argument<double> ox = parser.AddDoubleArg("-x");;
            Argument<bool> oy = parser.AddSwitchArg("-y");
            Argument<bool> on = parser.AddSwitchArg("-n");
            var choices = new string[]{"A", "B", "C"};
            var alignMethodNames = new Dictionary<AlignMethod, string> {{ AlignMethod.None,"none" }, { AlignMethod.Align,"align" }, { AlignMethod.Super,"super"}, { AlignMethod.Cealign,"cealign" }};
            Argument<string> oc = parser.AddStringArg("-c").SetChoices(choices).SetDefault("B");
            Argument<AlignMethod> od = parser.AddDictionaryChoiceArg(alignMethodNames, "-d").SetDefault(AlignMethod.Align);
            Argument<string> ars = parser.AddStringArg("STRING");
            Argument<string[]> arv = parser.AddArg(2, args=> args, "VARARGS");
            Argument<int> ari = parser.AddIntArg("INT");
            Argument<bool> ard = parser.AddSwitchArg("[SWITCH]");
            Argument<bool> ard2 = parser.AddSwitchArg("DUMMY?");

            bool optionsOK = parser.TryParse(args);
            if (!optionsOK) return false;
            
            foreach (var opt in parser.Arguments)
                Console.WriteLine($"{opt}");
            foreach (var opt in parser.Options)
                Console.WriteLine($"{opt}");
            return true;
        }
        
        public static void EnumerablePerformanceTest(){
            const int TRIALS = 3;
            const int ITERS = 1000_000;
            const int SIZE = 1_000;
            MyStopwatch watch = new MyStopwatch(Console.WriteLine);
            for (int i = 0; i < TRIALS; i++){
                watch.Start();
                for (int j = 0; j < ITERS; j++){
                    int[] arr = Enumerable.Range(0, SIZE).ToArray();
                    int x = arr[SIZE-1];
                }
                watch.Stop("Enumerable");
            }
            for (int i = 0; i < TRIALS; i++){
                watch.Start();
                for (int j = 0; j < ITERS; j++){
                    int[] arr = Range(0, SIZE);
                    int x = arr[SIZE-1];
                }
                watch.Stop("Range");
            }
        }

        private static int[] Range(int start, int end){
            if (start > end) throw new ArgumentException("End must be >= start.");
            int length = end - start;
            int[] result = new int[length];
            for (int i = 0; i < length; i++){
                result[i] = start + i;
            }
            return result;
        }
    }
    
    public class MyStopwatch {
        private DateTime t0;
        private Action<string> printFunction;

        public MyStopwatch(Action<string> printFunction){
            this.printFunction = printFunction;
            Start();
        }

        /** Reset to 0 */
        public void Start(){
            this.t0 = DateTime.Now;
        }

        /** Print the time since the last start/stop without reseting */
        public void Lap(string message){
            printFunction($"{message}: {DateTime.Now.Subtract(t0)}");
        }
        
        /** Print the time since the last start/stop and reset */
        public void Stop(string message){
            Lap(message);
            Start();
        }
    }
}
