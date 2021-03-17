using System;
using System.Linq;

namespace StructureCutter
{
    public class Domain
    {
        public string Name { get; private set; }
        public string Pdb { get; private set; }
        public string Chain { get; private set; }
        public string RangeString { get; private set; }
        public (int,int)[] Ranges { get; private set; }
        
        public Domain(string name, string pdb, string chain, string rangeString) {
            this.Name = name;
            this.Pdb = pdb;
            this.Chain = chain;
            this.RangeString = rangeString;
            this.Ranges = ParseRangeString(rangeString);
        }

        public override string ToString(){
            return $"Domain {Name} ({Pdb} {Chain} {RangeString})";
        }

        private static (int,int) ParseOneRangeString(string rangeString) {
            string[] parts = rangeString.Split(':').ToArray();
            if (parts.Length != 2){
                throw new FormatException($"Cannot parse range string: {rangeString}");
            } else {
                try {
                    int start = parts[0]=="" ? Int32.MinValue : Int32.Parse(parts[0]);
                    int end = parts[1]=="" ? Int32.MaxValue : Int32.Parse(parts[1]);
                    return (start, end);
                } catch {
                    throw new FormatException($"Cannot parse range string: {rangeString}");
                }
            }
        }
        public static (int,int)[] ParseRangeString(string rangeString) {
            return rangeString.Split(',').Select(range => ParseOneRangeString(range)).ToArray();
        }
    }
}