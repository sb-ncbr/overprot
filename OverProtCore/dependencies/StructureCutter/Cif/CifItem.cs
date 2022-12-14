
using System;
using System.Collections.Generic;
using System.Linq;
using Cif.Raw;

namespace Cif
{
    public class CifItem
    {
        private CifParser parser;
        internal int iTag; //TODO make private
        public string CategoryName { get; private set; }
        public string KeywordName { get; private set; }
        public string FullName { get; private set; }
        public int Count { get; private set; }

        // Not to be instantiated directly. Use CifCategory.GetItem() or CifBlock.GetItem().
        internal CifItem(CifParser parser, string keyword, int iTag){
            this.parser = parser;
            this.iTag = iTag;
            this.FullName = parser.TagNames[iTag];
            string [] parts = FullName.Split('.', 2);
            this.CategoryName = parts[0];
            this.KeywordName = parts[1];
            this.Count = parser.CountValuesForTag(iTag);
        }
        
        public object GetValues(CifValueType type){
            switch(type){
                case CifValueType.String:
                    return GetStrings();
                case CifValueType.Char:
                    return GetChars();
                case CifValueType.Integer:
                    return GetIntegers();
                case CifValueType.Double:
                    return GetDoubles();
                default:
                    throw new NotImplementedException();
            }
        }
        public object GetValues(CifValueType type, int[] rows){
            switch(type){
                case CifValueType.String:
                    return GetStrings(rows);
                case CifValueType.Char:
                    return GetChars(rows);
                case CifValueType.Integer:
                    return GetIntegers(rows);
                case CifValueType.Double:
                    return GetDoubles(rows);
                default:
                    throw new NotImplementedException();
            }
        }

        public string[] GetStrings () => parser.GetValuesAsStrings(iTag);
        public string[] GetStrings (int[] rows) => parser.GetValuesAsStrings(iTag, rows);
        
        public char[] GetChars () => parser.GetValuesAsChars(iTag);
        public char[] GetChars (int[] rows) => parser.GetValuesAsChars(iTag, rows);

        public int[] GetIntegers () => parser.GetValuesAsIntegers(iTag);
        public int[] GetIntegers (int[] rows) => parser.GetValuesAsIntegers(iTag, rows);
        public int[] GetIntegers (int defaultValue) => parser.GetValuesAsIntegers(iTag, defaultValue);
        public int[] GetIntegers (int[] rows, int defaultValue) => parser.GetValuesAsIntegers(iTag, rows, defaultValue);

        public double[] GetDoubles () => parser.GetValuesAsDoubles(iTag);
        public double[] GetDoubles (int[] rows) => parser.GetValuesAsDoubles(iTag, rows);
        public double[] GetDoubles (double defaultValue) => parser.GetValuesAsDoubles(iTag, defaultValue);
        public double[] GetDoubles (int[] rows, double defaultValue) => parser.GetValuesAsDoubles(iTag, rows, defaultValue);
        
        public bool[] GetTrueWhereFirstCharacterMatches (int[] rows, char testedCharacter) => parser.GetTrueWhereFirstCharacterMatches(iTag, rows, testedCharacter);

        /// <summary>
        /// Gets indices of rows which fulfil the given predicate "predicate".
        /// </summary>
        /// <param name="predicate"> Predicate function: takes string Value, returns true iff a row containing Value should be selected.</param>
        //public List<int> GetRowsWhere (Func<string,bool> predicate) => parser.GetIndicesWhere(iTag, predicate);

        /// <summary>
        /// Gets indices of rows which fulfil the given predicate "predicate".
        /// </summary>
        /// <param name="predicate"> Predicate function: 
        /// takes string Text (full text of CIF file), 
        /// int Start (index of the first character of the CIF value within Text), 
        /// int Stop (index of the last character of the CIF value within Text + 1),
        /// returns true iff a row containing this CIF value should be selected.</param>
        /// <remarks>
        /// Can be faster than calling GetIndicesWhere (Func<string,bool> predicate).
        /// Does not implicitly instantiate text.Substring(i, j).
        /// </remarks>
        // public List<int> GetRowsWhere (Func<string,int,int,bool> predicate) => parser.GetIndicesWhere(iTag, predicate);

        /// <summary>
        /// Gets indices of rows which contain exactly the string "sample".
        /// </summary>
        // public List<int> GetRowsWith (string sample) => parser.GetIndicesWith(iTag, sample);

        /// <summary>
        /// Gets indices of rows which contain any of the strings "samples".
        /// </summary>
        // public List<int> GetRowsWith (params string[] samples) => parser.GetIndicesWith(iTag, samples);
        ///
        
        public int[] GetRowsGroupedByValue(out int[] startsOfGroups) {
            return GetRowsGroupedByValue(Enumerable.Range(0, Count).ToArray(), out startsOfGroups);
        }
        public int[] GetRowsGroupedByValue(int[] rows, out int[] startsOfGroups) {
            int[] _;
            return parser.GroupByValuesInEachRegion(iTag, rows, new int[]{0, rows.Length}, out startsOfGroups, out _);
        }

        public int[] GetRowsGroupedByValueInEachRegion(int[] rows, int[] startsOfRegions, out int[] startsOfGroups, out int[] startGroupsOfRegions) {
            return parser.GroupByValuesInEachRegion(iTag, rows, startsOfRegions, out startsOfGroups, out startGroupsOfRegions);
        }
    }
}