using System.Collections.Generic;

namespace StructureCutter
{
    public class ChainSummary
    {
        public string chain { get; set; }       // _atom_site.label_asym_id
        public string auth_chain { get; set; }  // _atom_site.auth_asym_id
        public int n_atoms { get; set; }        // number of non-hydrogen atoms
        public double[] center { get; set; }    // average XYZ coordinates of non-hydrogen atoms

        public string entity { get; set; }               // _entity.id / _atom_site.label_entity_id
        public string entity_type { get; set; }          // _entity.type
        public string entity_polymer_type { get; set; }  // _entity_poly.type
        public string entity_comp { get; set; }          // _pdbx_entity_nonpoly.comp_id 
        public string entity_description { get; set; }   // _entity.pdbx_description

        public List<ResidueSummary> residues { get; set; } // residue summaries

        public override string ToString()
        {
            double x = center[0];
            double y = center[1];
            double z = center[2];
            return $"{chain} [auth {auth_chain}]\t{n_atoms}\t[{x,8:F3}, {y,8:F3}, {z,8:F3}]\tentity {entity}\t{entity_type,-11}\t{entity_polymer_type??".",-19}\t{entity_comp??"."}\t{entity_description}";
        }
    }

}