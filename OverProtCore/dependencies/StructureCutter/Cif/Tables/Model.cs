using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Cif.Libraries;

namespace Cif.Tables
{
    public class Model
    {
        public const string MODEL_NUM_COLUMN = "pdbx_PDB_model_num";
        public const int DEFAULT_MODEL_NUM = 0;

        public int ModelNumber { get; private set; }
        public AtomTable Atoms { get; private set; }
        public ResidueTable Residues { get; private set; }
        public FragmentTable Fragments { get; private set; }
        public ChainTable Chains { get; private set; }
        public EntityTable Entities { get; private set; }

        ///<summary> Not to be called directly! Use ModelCollection.GetModel() or similar.</summary>
        internal Model(CifCategory category, int[] rows, int modelNumber){
            this.ModelNumber = modelNumber;

            // entities
            CifItem entityId = category[EntityTable.KEY_COLUMN];
            int[] atomStartsOfEntities;
            rows = entityId.GetRowsGroupedByValue(rows, out atomStartsOfEntities);

            // chains
            CifItem asymId = category[ChainTable.KEY_COLUMN];
            int[] atomStartsOfChains;
            int[] chainStartsOfEntities;
            rows = asymId.GetRowsGroupedByValueInEachRegion(rows, atomStartsOfEntities, out atomStartsOfChains, out chainStartsOfEntities);

            // fragments + residues
            CifItem residueId = category[ResidueTable.KEY_COLUMN];
            int[] fragmentStartsOfChains;
            int[] residueStartsOfFragments;
            int[] atomStartsOfResidues;
            int[] residueNumbers;
            GetFragmentsAndResidues(residueId, ref rows, atomStartsOfChains, out fragmentStartsOfChains, out residueStartsOfFragments, out atomStartsOfResidues, out residueNumbers);
            
            // remaining combinations
            int[] residueStartsOfChains = GetSelectedElements(residueStartsOfFragments, fragmentStartsOfChains, false);
            int[] residueStartsOfEntities = GetSelectedElements(residueStartsOfChains, chainStartsOfEntities, false);
            int[] fragmentStartsOfEntities = GetSelectedElements(fragmentStartsOfChains, chainStartsOfEntities, false);
            int[] atomStartsOfFragments = GetSelectedElements(atomStartsOfResidues, residueStartsOfFragments, false);

            // fill fields
            this.Entities = new EntityTable(this, category, rows, atomStartsOfEntities, residueStartsOfEntities, fragmentStartsOfEntities, chainStartsOfEntities);
            this.Chains = new ChainTable(this, category, rows, atomStartsOfChains, residueStartsOfChains, fragmentStartsOfChains, chainStartsOfEntities);
            this.Fragments = new FragmentTable(this, category, rows, atomStartsOfFragments, residueStartsOfFragments, fragmentStartsOfChains, fragmentStartsOfEntities);
            this.Residues = new ResidueTable(this, category, rows, atomStartsOfResidues, residueStartsOfFragments, residueStartsOfChains, residueStartsOfEntities);
            this.Atoms = new AtomTable(this, category, rows, atomStartsOfResidues, atomStartsOfFragments, atomStartsOfChains, atomStartsOfEntities);
        }

        private static void GetFragmentsAndResidues(
            CifItem residueNumberItem, 
            ref int[] rows, 
            int[] atomStartsOfChains, 
            out int[] fragmentStartsOfChains,
            out int[] residueStartsOfFragments,
            out int[] atomStartsOfResidues,
            out int[] residueNumbersOfResidues)
        {
            if (rows.Length == 0){
                Lib.WriteWarning("Creating model from 0 atoms.");
            }
            int[] residueNumbers = residueNumberItem.GetIntegers(rows, ResidueTable.DEFAULT_RESIDUE_NUMBER);
            int nChains = atomStartsOfChains.Length - 1; // the last is sentinel
            List<int> fragOfChainList = new List<int>();
            List<int> resOfFragList = new List<int>();
            List<int> atomOfResList = new List<int>();
            int iResidue = 0;
            int iFragment = 0;
            for (int iChain = 0; iChain < nChains; iChain++) {
                int startAtom = atomStartsOfChains[iChain];
                int endAtom = atomStartsOfChains[iChain+1];
                // start new residue, new fragment, new chain
                int resNum = residueNumbers[startAtom];
                atomOfResList.Add(startAtom);      
                resOfFragList.Add(iResidue);   
                fragOfChainList.Add(iFragment);
                for (int iAtom = startAtom+1; iAtom < endAtom; iAtom++) {
                    int newResNum = residueNumbers[iAtom];
                    if (newResNum == resNum) {
                        // continue residue (do nothing)
                    } else if (newResNum == resNum + 1) {
                        // new residue, continue fragment
                        //TODO start new residue if this is water
                        resNum = newResNum;
                        atomOfResList.Add(iAtom);
                        iResidue++;
                    } else if (newResNum > resNum) {
                        // new residue, new fragment
                        resNum = newResNum;
                        atomOfResList.Add(iAtom);
                        iResidue++;
                        resOfFragList.Add(iResidue);
                        iFragment++;
                    } else {
                        // decreasing residue number => throw exception
                        throw new NotImplementedException("Residue numbers (" + residueNumberItem.FullName + ") are not in increasing order (not supported by current version)");
                    }
                }
                // finish residue, fragment, chain
                iResidue++;
                iFragment++;
            }
            fragmentStartsOfChains = Lib.AppendAndCopyToArray(fragOfChainList, iFragment); // fragOfChainList.Append(iFragment).ToArray();
            residueStartsOfFragments = Lib.AppendAndCopyToArray(resOfFragList, iResidue); // resOfFragList.Append(iResidue).ToArray();
            atomStartsOfResidues = Lib.AppendAndCopyToArray(atomOfResList, atomStartsOfChains[nChains]); // atomOfResList.Append(atomStartsOfChains[nChains]).ToArray();
            residueNumbersOfResidues = atomOfResList.Select(a => residueNumbers[a]).ToArray();
        }

        internal static int[] GetSelectedElements(int[] elements, int[] indices, bool ignoreLastIndex) {
            int nSelected = ignoreLastIndex ? indices.Length - 1 : indices.Length;
            int[] selected = new int[nSelected];
            for (int i = 0; i < nSelected; i++)
            {
                selected[i] = elements[indices[i]];
            }
            return selected;
        }

        internal static int[] GetUpRefs(int[] downRefs){ 
            int nTop = downRefs.Length - 1;
            int nBottom = downRefs[nTop];
            int[] upRefs = new int[nBottom];
            for (int iTop = 0; iTop < nTop; iTop++) {
                for (int iBottom = downRefs[iTop]; iBottom < downRefs[iTop+1]; iBottom++) {
                    upRefs[iBottom] = iTop;
                }
            }
            return upRefs;
        }

        public string Print() {
            const string IN = "    ";
            StringBuilder builder = new StringBuilder();
            for (int iEntity = 0; iEntity < Entities.Count; iEntity++) {
                builder.Append("Entity " + Entities.String(iEntity) + "\n");
                for (int iChain = Entities.ChainStartIndex[iEntity]; iChain < Entities.ChainEndIndex[iEntity]; iChain++) {
                    builder.Append(IN + "Chain " + Chains.String(iChain) + "\n");
                    for (int iFragment = Chains.FragmentStartIndex[iChain]; iFragment < Chains.FragmentEndIndex[iChain]; iFragment++) {
                        builder.Append(IN + IN + "Fragment " + Fragments.String(iFragment) + "\n");
                        for (int iResidue = Fragments.ResidueStartIndex[iFragment]; iResidue < Fragments.ResidueEndIndex[iFragment]; iResidue++) {
                            builder.Append(IN + IN + IN + "Residue " + Residues.String(iResidue) + "\n");
                            for (int iAtom = Residues.AtomStartIndex[iResidue]; iAtom < Residues.AtomEndIndex[iResidue]; iAtom++) {
                                builder.Append(IN + IN + IN + IN + "Row " + Atoms.RowIndex[iAtom] + ":   Atom " + Atoms.String(iAtom) + "\n");
                            }
                        }
                    }
                }
            }
            return builder.ToString();
        }
    }
}