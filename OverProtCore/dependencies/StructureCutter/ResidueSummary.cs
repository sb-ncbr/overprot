
namespace StructureCutter
{
    public class ResidueSummary
    {
        public int? residue { get; set; }       // _atom_site.label_seq_id
        public int? auth_residue { get; set; }  // _atom_site.auth_seq_id
        public string auth_ins_code { get; set; }  // _atom_site.pdbx_PDB_ins_code
        public string comp { get; set; }  // _atom_site.label_comp_id
        public int n_atoms { get; set; }        // number of non-hydrogen atoms
        public double[] center { get; set; }    // average XYZ coordinates of non-hydrogen atoms
        public double[] trace { get; set; }    // average XYZ coordinates of non-hydrogen atoms
        public double[] tip { get; set; }    // average XYZ coordinates of non-hydrogen atoms
        // add position of CA atom / some atom from DNA (name P)
        // add position of base tip atom from DNA? (C, T, U, DC, DT, DU: N3; A, G, DA, DG: N1)

        public override string ToString()
        {
            double x = center[0];
            double y = center[1];
            double z = center[2];
            return $"{residue} [auth {auth_residue}]\t{n_atoms}\t{comp??"."}";
        }
    }

}