using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

using Cif;
using Cif.Filtering;

namespace StructureCutter
{
    public class StructureSummary
    {
        public const int XYZ_DECIMAL_DIGITS = 3;

        public readonly string Name;
        public readonly Dictionary<string, ChainSummary> ChainSummaries;
        public readonly int NAtomsAll;
        public readonly int NAtomsFiltered;

        public StructureSummary(string name, CifBlock block)
        {
            Name = name;
            CifCategory _atom_site = block.GetCategory("_atom_site");
            NAtomsAll = _atom_site.RowCount;
            // Console.WriteLine($"\n{name} [{_atom_site.RowCount}]");

            Filter filter = !Filter.StringEquals("type_symbol", "H");

            // Select only the first model, if there are more than one
            if (_atom_site.ContainsItem("pdbx_PDB_model_num")){
                int[] modelNumbers = _atom_site.GetItem("pdbx_PDB_model_num").GetIntegers().Distinct().ToArray();
                if (modelNumbers.Length == 0) {
                    Console.Error.WriteLine($"Warning: No model found for structure {Name}");
                    throw new Exception($"No model found for structure {Name}");
                }
                if (modelNumbers.Length > 1) {
                    int firstModelNumber = modelNumbers[0];
                    Console.Error.WriteLine($"Warning: More than one model found for structure {Name} (using only model {firstModelNumber})");
                    Filter modelFilter = Filter.IntegerInRange("pdbx_PDB_model_num", (firstModelNumber, firstModelNumber));
                    filter &= modelFilter;
                }
            }

            int[] rows = filter.GetFilteredRows(_atom_site).ToArray();
            NAtomsFiltered = rows.Length;

            // Atoms
            string[] atom_Chain = _atom_site.GetItem("label_asym_id").GetStrings(rows);
            string[] atom_AuthChain = _atom_site.GetItem("auth_asym_id").GetStrings(rows);
            string[] atom_Entity = _atom_site.GetItem("label_entity_id").GetStrings(rows);
            double[] atom_X = _atom_site.GetItem("Cartn_x").GetDoubles(rows);
            double[] atom_Y = _atom_site.GetItem("Cartn_y").GetDoubles(rows);
            double[] atom_Z = _atom_site.GetItem("Cartn_z").GetDoubles(rows);

            var indexDict = new Dictionary<string, List<int>>();  // Maps chain to list of atom indices
            for (int i = 0; i < atom_Chain.Length; i++) {
                string chain = atom_Chain[i];
                if (!indexDict.ContainsKey(chain)){
                    indexDict[chain] = new List<int>();
                }
                indexDict[chain].Add(i);
            }

            // Entities
            CifCategory _entity = block.GetCategory("_entity");
            string[] entity_Id = _entity.GetItem("id").GetStrings();
            string[] entity_Type = _entity.GetItem("type").GetStrings();
            string[] entity_Description = _entity.GetItem("pdbx_description").GetStrings();
            var entity_TypeDict = ZipDict(entity_Id, entity_Type);
            var entity_DescriptionDict = ZipDict(entity_Id, entity_Description);

            // _entity.type:
            // branched	
            // macrolide	
            // non-polymer	
            // polymer	
            // water

            Dictionary<string,string> entity_CompDict;
            if (block.ContainsCategory("_pdbx_entity_nonpoly")){
                CifCategory _pdbx_entity_nonpoly = block.GetCategory("_pdbx_entity_nonpoly");
                string[] pen_Id = _pdbx_entity_nonpoly.GetItem("entity_id").GetStrings();
                string[] pen_CompId = _pdbx_entity_nonpoly.GetItem("comp_id").GetStrings();
                entity_CompDict = ZipDict(pen_Id, pen_CompId);
            } else {
                entity_CompDict = new Dictionary<string, string>();
            }
            Dictionary<string,string> entity_PolyTypeDict;
            if (block.ContainsCategory("_entity_poly")){
                CifCategory _entity_poly = block.GetCategory("_entity_poly");
                string[] ep_Id = _entity_poly.GetItem("entity_id").GetStrings();
                string[] ep_PolyType = _entity_poly.GetItem("type").GetStrings();
                entity_PolyTypeDict = ZipDict(ep_Id, ep_PolyType);
            } else {
                entity_PolyTypeDict = new Dictionary<string, string>();
            }
            ChainSummaries = new Dictionary<string, ChainSummary>();
            foreach (string chain in indexDict.Keys) {
                ChainSummary sum = new ChainSummary();
                sum.chain = chain;
                List<int> indices = indexDict[chain];
                sum.auth_chain = TheOnlyValue(atom_AuthChain, indices, errorMessage: $"{name}: label_asym_id '{chain}' maps to multiple auth_asym_ids.");
                sum.entity = TheOnlyValue(atom_Entity, indices, errorMessage: $"{name}: label_asym_id '{chain}' maps to multiple label_entity_ids.");
                sum.n_atoms = indexDict[chain].Count();
                double x = Math.Round(indices.Select(i => atom_X[i]).Average(), XYZ_DECIMAL_DIGITS);
                double y = Math.Round(indices.Select(i => atom_Y[i]).Average(), XYZ_DECIMAL_DIGITS);
                double z = Math.Round(indices.Select(i => atom_Z[i]).Average(), XYZ_DECIMAL_DIGITS);
                sum.center = new double[]{x, y, z};
                sum.entity_type = entity_TypeDict[sum.entity];
                sum.entity_description = entity_DescriptionDict[sum.entity];
                sum.entity_polymer_type = entity_PolyTypeDict.ContainsKey(sum.entity) ? entity_PolyTypeDict[sum.entity] : null;
                sum.entity_comp = entity_CompDict.ContainsKey(sum.entity) ? entity_CompDict[sum.entity] : null;
                ChainSummaries[chain] = sum;
            }
            // TODO test on NMR structure if really filters out H and multiple models1

            // TODO _entity_name_com.name, _pdbx_entity_nonpoly.name ?
        }

        public override string ToString(){
            var lines = new List<string>(ChainSummaries.Count + 1);
            lines.Add($"\n{Name} [{NAtomsAll} -> {NAtomsFiltered} atoms]");
            foreach (var sum in ChainSummaries.Values){
                lines.Add(sum.ToString());
            }
            return string.Join("\n\t", lines);
        }

        public void Save(string filename){
            using (StreamWriter w = new StreamWriter(filename)){
                w.WriteLine(System.Text.Json.JsonSerializer.Serialize(ChainSummaries));                
            }
        }

        private static T TheOnlyValue<T>(IList<T> items, IEnumerable<int> indices, T defaultValue = default(T), string errorMessage = "items contains more that one distinct value.") {
            T[] distinctValues = indices.Select(i => items[i]).Distinct().ToArray();
            if (distinctValues.Length == 0) {
                return defaultValue;
            } else if (distinctValues.Length == 1) {
                return distinctValues[0];
            } else {
                throw new ArgumentException(errorMessage);
            }
        }

        private static Dictionary<K,V> ZipDict<K,V>(IList<K> keys, IList<V> values) {
            if (keys.Count != values.Count) {
                throw new ArgumentException("Keys and values must have the same length");
            }
            var result = new Dictionary<K,V>(keys.Count);
            for (int i = 0; i < keys.Count; i++){
                result[keys[i]] = values[i];
            }
            return result;
        }

    }
}