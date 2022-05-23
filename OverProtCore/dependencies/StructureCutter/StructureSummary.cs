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
        const int RESI_NULL = int.MinValue;
        static readonly Dictionary<string, string> BASE_TIP_NAMES = BaseTipNames();

        public readonly string Name;
        public readonly Dictionary<string, ChainSummary> ChainSummaries;
        public readonly int NAtomsAll;
        public readonly int NAtomsFiltered;

        public StructureSummary(string name, CifBlock block, bool includeResidueSummaries, int[] rows = null)
        {
            Name = name;
            CifCategory _atom_site = block.GetCategory("_atom_site");
            if (rows != null)
            {
                NAtomsAll = rows.Length;
            }
            else
            {
                NAtomsAll = _atom_site.RowCount;
                rows = Enumerable.Range(0, NAtomsAll).ToArray();
            }

            Filter filter = Filter.TheseRows(rows) & Filter.StringEquals("label_alt_id", new string[]{".", "A"}) & !Filter.StringEquals("type_symbol", "H");
            rows = filter.GetFilteredRows(_atom_site).ToArray();
            NAtomsFiltered = rows.Length;

            // Atoms
            string[] atom_Entity = _atom_site.GetItem("label_entity_id").GetStrings(rows);
            string[] atom_Chain = _atom_site.GetItem("label_asym_id").GetStrings(rows);
            string[] atom_AuthChain = _atom_site.GetItem("auth_asym_id").GetStrings(rows);
            int[] atom_Residue = null;
            int[] atom_AuthResidue = null;
            string[] atom_AuthInsCode = null;
            string[] atom_Compound = null;
            string[] atom_Name = null;
            if (includeResidueSummaries)
            {
                atom_Residue = _atom_site.GetItem("label_seq_id").GetIntegers(rows, RESI_NULL);
                atom_AuthResidue = _atom_site.ContainsItem("auth_seq_id") ?
                    _atom_site.GetItem("auth_seq_id").GetIntegers(rows, RESI_NULL)
                    : NewFullArray(NAtomsFiltered, RESI_NULL);
                atom_AuthInsCode = _atom_site.ContainsItem("pdbx_PDB_ins_code") ?
                    _atom_site.GetItem("pdbx_PDB_ins_code").GetStrings(rows)
                    : NewFullArray(NAtomsFiltered, "");
                atom_AuthInsCode.Replace("?", "");
                atom_AuthInsCode.Replace(".", "");
                atom_Compound = _atom_site.GetItem("label_comp_id").GetStrings(rows);
                atom_Name = _atom_site.GetItem("label_atom_id").GetStrings(rows);
            }
            double[] atom_X = _atom_site.GetItem("Cartn_x").GetDoubles(rows);
            double[] atom_Y = _atom_site.GetItem("Cartn_y").GetDoubles(rows);
            double[] atom_Z = _atom_site.GetItem("Cartn_z").GetDoubles(rows);

            var chainIndex = new Dictionary<string, List<int>>();  // Maps chain to list of atom indices
            var chainResidueIndex = new Dictionary<string, Dictionary<(int resi, int authResi, string authIns), List<int>>>();  // Maps chain and residue to list of atom indices
            if (includeResidueSummaries)
            {
                for (int i = 0; i < NAtomsFiltered; i++)
                {
                    string chain = atom_Chain[i];
                    var residue = (atom_Residue[i], atom_AuthResidue[i], atom_AuthInsCode[i]);
                    if (!chainIndex.ContainsKey(chain))
                    {
                        chainIndex[chain] = new List<int>();
                        chainResidueIndex[chain] = new Dictionary<(int, int, string), List<int>>();
                    }
                    if (!chainResidueIndex[chain].ContainsKey(residue))
                    {
                        chainResidueIndex[chain][residue] = new List<int>();
                    }
                    chainIndex[chain].Add(i);
                    chainResidueIndex[chain][residue].Add(i);
                }
            }
            else
            {
                for (int i = 0; i < NAtomsFiltered; i++)
                {
                    string chain = atom_Chain[i];
                    if (!chainIndex.ContainsKey(chain))
                    {
                        chainIndex[chain] = new List<int>();
                    }
                    chainIndex[chain].Add(i);
                }
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

            Dictionary<string, string> entity_CompDict;
            if (block.ContainsCategory("_pdbx_entity_nonpoly"))
            {
                CifCategory _pdbx_entity_nonpoly = block.GetCategory("_pdbx_entity_nonpoly");
                string[] pen_Id = _pdbx_entity_nonpoly.GetItem("entity_id").GetStrings();
                string[] pen_CompId = _pdbx_entity_nonpoly.GetItem("comp_id").GetStrings();
                entity_CompDict = ZipDict(pen_Id, pen_CompId);
            }
            else
            {
                entity_CompDict = new Dictionary<string, string>();
            }
            Dictionary<string, string> entity_PolyTypeDict;
            if (block.ContainsCategory("_entity_poly"))
            {
                CifCategory _entity_poly = block.GetCategory("_entity_poly");
                string[] ep_Id = _entity_poly.GetItem("entity_id").GetStrings();
                string[] ep_PolyType = _entity_poly.GetItem("type").GetStrings();
                entity_PolyTypeDict = ZipDict(ep_Id, ep_PolyType);
            }
            else
            {
                entity_PolyTypeDict = new Dictionary<string, string>();
            }
            ChainSummaries = new Dictionary<string, ChainSummary>();
            foreach (string chain in chainIndex.Keys)
            {
                ChainSummary sum = new ChainSummary();
                List<int> indices = chainIndex[chain];
                sum.chain = chain;
                sum.auth_chain = TheOnlyValue(atom_AuthChain, indices, errorMessage: $"{name}: label_asym_id '{chain}' maps to multiple auth_asym_ids.");
                sum.entity = TheOnlyValue(atom_Entity, indices, errorMessage: $"{name}: label_asym_id '{chain}' maps to multiple label_entity_ids.");
                sum.n_atoms = indices.Count;
                sum.center = Center(atom_X, atom_Y, atom_Z, indices);
                sum.entity_type = entity_TypeDict[sum.entity];
                sum.entity_description = entity_DescriptionDict[sum.entity];
                sum.entity_polymer_type = entity_PolyTypeDict.ContainsKey(sum.entity) ? entity_PolyTypeDict[sum.entity] : null;
                sum.entity_comp = entity_CompDict.ContainsKey(sum.entity) ? entity_CompDict[sum.entity] : null;
                ChainSummaries[chain] = sum;
                if (includeResidueSummaries && (sum.entity_type == "polymer" || sum.entity_type == "branched"))
                {
                    sum.residues = new List<ResidueSummary>();
                    foreach (var residue in chainResidueIndex[chain].Keys)
                    {
                        ResidueSummary ressum = new ResidueSummary();
                        indices = chainResidueIndex[chain][residue];

                        ressum.residue = residue.resi != RESI_NULL ? residue.resi : null;
                        ressum.auth_residue = residue.authResi != RESI_NULL ? residue.authResi : null;
                        ressum.auth_ins_code = residue.authIns;
                        ressum.comp = TheOnlyValue(atom_Compound, indices, errorMessage: $"{name}: label_seq_id '{ressum.residue}' maps to multiple label_comp_id.");
                        ressum.n_atoms = indices.Count;
                        ressum.center = Center(atom_X, atom_Y, atom_Z, indices);
                        if (BASE_TIP_NAMES.ContainsKey(ressum.comp)) // is a nucleic base?
                        {
                            ressum.trace = AtomPosition("P", atom_Name, atom_X, atom_Y, atom_Z, indices);
                            ressum.tip = AtomPosition(BASE_TIP_NAMES[ressum.comp], atom_Name, atom_X, atom_Y, atom_Z, indices);
                        }
                        else
                        {
                            ressum.trace = AtomPosition("CA", atom_Name, atom_X, atom_Y, atom_Z, indices);
                        }
                        sum.residues.Add(ressum);
                    }
                }
            }

            // TODO _entity_name_com.name, _pdbx_entity_nonpoly.name ?
        }

        private static double[] Center(double[] xs, double[] ys, double[] zs, List<int> indices)
        {
            double x = Math.Round(indices.Select(i => xs[i]).Average(), XYZ_DECIMAL_DIGITS);
            double y = Math.Round(indices.Select(i => ys[i]).Average(), XYZ_DECIMAL_DIGITS);
            double z = Math.Round(indices.Select(i => zs[i]).Average(), XYZ_DECIMAL_DIGITS);
            return new double[] { x, y, z };
        }

        private static double[] AtomPosition(string atomName, string[] atomNames, double[] xs, double[] ys, double[] zs, List<int> indices)
        {
            int index = indices.FirstOrDefault(i => atomNames[i] == atomName, -1);
            if (index == -1) return null;
            double x = Math.Round(xs[index], XYZ_DECIMAL_DIGITS);
            double y = Math.Round(ys[index], XYZ_DECIMAL_DIGITS);
            double z = Math.Round(zs[index], XYZ_DECIMAL_DIGITS);
            return new double[] { x, y, z };
        }

        private static Dictionary<string, string> BaseTipNames()
        {
            Dictionary<string, string> result = new Dictionary<string, string>();
            result["A"] = "N1";
            result["G"] = "N1";
            result["DA"] = "N1";
            result["DA"] = "N1";
            result["C"] = "N3";
            result["T"] = "N3";
            result["U"] = "N3";
            result["DC"] = "N3";
            result["DT"] = "N3";
            result["DU"] = "N3";
            result["5CM"] = "N3";  // methylated cytosin
            // TODO what about methylated cytosin in RNA?
            return result;
        }

        public void DropResidueSummaries()
        {
            foreach (var chainSummary in ChainSummaries.Values)
            {
                chainSummary.residues = null;
            }
        }

        public override string ToString()
        {
            var lines = new List<string>(ChainSummaries.Count + 1);
            lines.Add($"\n{Name} [{NAtomsAll} -> {NAtomsFiltered} atoms]");
            foreach (var sum in ChainSummaries.Values)
            {
                lines.Add(sum.ToString());
            }
            return string.Join("\n\t", lines);
        }

        public void Save(string filename)
        {
            using (StreamWriter w = new StreamWriter(filename))
            {
                w.WriteLine(System.Text.Json.JsonSerializer.Serialize(ChainSummaries));
            }
        }

        private static T TheOnlyValue<T>(IList<T> items, IEnumerable<int> indices, T defaultValue = default(T), string errorMessage = "items contains more that one distinct value.")
        {
            T[] distinctValues = indices.Select(i => items[i]).Distinct().ToArray();
            if (distinctValues.Length == 0)
            {
                return defaultValue;
            }
            else if (distinctValues.Length == 1)
            {
                return distinctValues[0];
            }
            else
            {
                Console.WriteLine("--> " + string.Join(", ", distinctValues)); //DEBUG
                throw new ArgumentException(errorMessage);
            }
        }

        private static Dictionary<K, V> ZipDict<K, V>(IList<K> keys, IList<V> values)
        {
            if (keys.Count != values.Count)
            {
                throw new ArgumentException("Keys and values must have the same length");
            }
            var result = new Dictionary<K, V>(keys.Count);
            for (int i = 0; i < keys.Count; i++)
            {
                result[keys[i]] = values[i];
            }
            return result;
        }

        private static T[] NewFullArray<T>(int length, T fillValue)
        {
            T[] result = new T[length];
            Array.Fill(result, fillValue);
            return result;
        }

    }
}