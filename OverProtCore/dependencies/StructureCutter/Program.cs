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
    public class Program
    {   
        const string DEFAULT_URL_SOURCE = "http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif";
        static readonly string[] DEFAULT_URL_SOURCES = { "http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif" };
        const string OUTPUT_EXTENSION = ".cif";
        const string PDB_OUTPUT_EXTENSION = ".pdb";
        const string SUMMARY_OUTPUT_EXTENSION = ".json";
        const char DEFAULT_CHAIN_ID = '.';  // Assigned to chains with multi-character IDs, when converting to PDB.
        const int FIRST_RESIDUE_ID = 1;  // Assigned to the first residue, when converting to PDB.
        const int SKIPPED_RESIDUES_AT_GAP = 1;  // Number of residue numbers to be skipped when some residues are missing in a chain, when converting to PDB.
        
        enum AlignMethod { None, Align, Super, Cealign };//DEBUG

        private class MainOptions : Options{
            StringOption os;
            public MainOptions(){
                os = AddStringOption("-s");
                (int i, string s) t = (x: 5, i: "ahoj");
                t.i = 5;
            }
        }
        private (string x, string y) GetArguments(string[] args){
            return (x: "ahoj", y: "cau");
        }

        static int Main(string[] args)
        {
            Options options = new Options();
            options.GlobalHelp = "StructureCutter 0.9";
            var optDomainListFile = options.AddStringOption("DOMAIN_LIST_FILE")
                .AddHelp("JSON file with the list of domains to be downloaded, in format")
                .AddHelp("[{\"pdb\": pdb, \"domain\": domain_name, \"chain_id\": chain_id, \"ranges\": ranges}*]")
                .AddHelp("or")
                .AddHelp("[[pdb, domain_name, chain_id, ranges]*]");

            var optCifOutdir = options.AddStringOption("--cif_outdir")
                .AddParameter("DIRECTORY")
                .AddHelp("Directory for output files in mmCIF format (or empty string to suppress mmCIF output).");
            var optPdbOutdir = options.AddStringOption("--pdb_outdir")
                .AddParameter("DIRECTORY")
                .AddHelp("Directory for output files in PDB format (or empty string to suppress PDB output).")
                .AddHelp("The PDB files will contain renumbered chain ID, residue number, and atom ID!");
            var optSummaryOutdir = options.AddStringOption("--summary_outdir")
                .AddParameter("DIRECTORY")
                .AddHelp("Directory for output files with chain summary (list of chains with their type, average XYZ, etc.)");
            var optResidueSummaryOutdir = options.AddStringOption("--residue_summary_outdir")
                .AddParameter("DIRECTORY")
                .AddHelp("Directory for output files with residue summary (same as chain summary plus info per residue)");
            var optSources = options.AddOption(1, args => args[0].Split(' ', StringSplitOptions.RemoveEmptyEntries), "--sources")
                .SetDefault(DEFAULT_URL_SOURCES)
                .AddParameter("SOURCES")
                .AddHelp("Space-separated list of URL sources (each starting with http:// or file:///).")
                .AddHelp("{pdb} in each source will be replaced by the actual PDB ID,")
                .AddHelp("{pdb_0}, {pdb_1}, {pdb_2}, {pdb_3} will be replaced by the individual characters of the PDB ID.")
                .AddHelp($"Default: '{string.Join(' ', DEFAULT_URL_SOURCES)}'");
            var optFailures = options.AddStringOption("--failures")
                .AddParameter("FILE")
                .AddHelp("Filename for output of failed PDBIDs. Use '-' to output on stderr. Default is to exit on any failures.");

            StringOption os = options.AddStringOption("-s");
            os.AddAction(() => Console.WriteLine(os.Value));
            IntOption oi = options.AddIntOption("-i").SetDefault(-1);
            oi.AddAction(() => Console.WriteLine(oi.Value));
            DoubleOption ox = options.AddDoubleOption("-x");
            SwitchOption oy = options.AddSwitchOption("-y");
            SwitchOption on = options.AddSwitchOption("-n");
            var choices = new string[]{"A", "B", "C"};
            var alignMethodNames = new Dictionary<AlignMethod, string> {{ AlignMethod.None,"none" }, { AlignMethod.Align,"align" }, { AlignMethod.Super,"super"}, { AlignMethod.Cealign,"cealign" }};
            ChoiceOption oc = options.AddChoiceOption(choices, "-c").SetDefault("B");
            DictionaryChoiceOption<AlignMethod> od = options.AddDictionaryChoiceOption(alignMethodNames, "-d").SetDefault(AlignMethod.Align);
            
            StringOption ars = options.AddStringOption("STRING");
            Option<string[]> arv = options.AddOption(2, args=> args, "VARARGS");
            IntOption ari = options.AddIntOption("INT");
            SwitchOption ard = options.AddSwitchOption("[SWITCH]");
            SwitchOption ard2 = options.AddSwitchOption("DUMMY?");

            bool optionsOK = options.TryParse(args);
            if (!optionsOK){
                return 1;
            }
            string domainListFile = optDomainListFile.Value;            
            string cifOutDirectory = optCifOutdir.Value;
            string pdbOutDirectory = optPdbOutdir.Value;
            string summaryOutDirectory = optSummaryOutdir.Value;
            string residueSummaryOutDirectory = optResidueSummaryOutdir.Value;
            string[] sources = optSources.Value;
            string failuresFile = optFailures.Value;

            Console.WriteLine($"OPTIONS: summaryOutDirectory: {summaryOutDirectory}");
            Console.WriteLine($"OPTIONS: {os} {os.Value}");
            Console.WriteLine($"OPTIONS: {oi} {oi.Value}");
            Console.WriteLine($"OPTIONS: {ox} {ox.Value}");
            Console.WriteLine($"OPTIONS: {oy} {oy.Value}");
            Console.WriteLine($"OPTIONS: {on} {on.Value}");
            Console.WriteLine($"OPTIONS: {oc} {oc.Value}");
            Console.WriteLine($"OPTIONS: {od} {od.Value}");
            Console.WriteLine($"OPTIONS: {optSources} {optSources.Value}" + string.Join(',', optSources.Value));
            Console.WriteLine($"{optSources} " + string.Join(',', sources));
            Console.WriteLine();
            // return 0;

            if (sources.Length == 0){
                Options.PrintError("Option --sources has invalid value. You must specify at least one source.");
                return 1;
            }
            foreach (string source in sources){
                string dummyUrl = FormatSource(source, "1234");
                if (!Uri.IsWellFormedUriString(dummyUrl, UriKind.RelativeOrAbsolute)) {
                    Options.PrintError($"Option --sources has invalid value. Invalid source: \"{source}\". The source must be a valid URL, possibly with special marks ({{pdb}}, {{pdb_0}}, {{pdb_1}}, {{pdb_2}}, {{pdb_3}}).");
                    return 1;
                }
            }

            if (cifOutDirectory == null && pdbOutDirectory == null && summaryOutDirectory == null){
                Console.WriteLine("WARNING: You did not specify any of --cif_outdir, --pdb_outdir, --summary_outdir. No output will be produced.");
            }
            

            Console.WriteLine($"Domain list file: {domainListFile}");
            Console.WriteLine($"Sources:");
            foreach(string source in sources){
                Console.WriteLine($"\t{source}");
            }
            Console.WriteLine($"CIF output directory: {cifOutDirectory}");
            Console.WriteLine($"PDB output directory: {pdbOutDirectory}");


            Domain[] domains;
            try {
                domains = Domain.ReadDomainsFromJson(domainListFile);
            } catch (IOException){
                Console.Error.WriteLine($"ERROR: Cannot find or read the domain list file '{domainListFile}'");
                return 2;
            } catch (FormatException ex){
                Console.Error.WriteLine($"ERROR: Domain list file has invalid format:");
                Console.Error.WriteLine($"       {ex.Message}");
                return 2;
            }
            Dictionary<string,List<Domain>> domainsByPdb = Domain.SortDomainsByPdb(domains);

            if (cifOutDirectory != null){
                Directory.CreateDirectory(cifOutDirectory);
            }
            if (pdbOutDirectory != null){
                Directory.CreateDirectory(pdbOutDirectory);
            }
            if (summaryOutDirectory != null){
                Directory.CreateDirectory(summaryOutDirectory);
            }
            if (failuresFile != null && failuresFile != "-"){
                using(var r = new StreamWriter(failuresFile, false)){
                    // clear file
                }
            }

            WebClient webClient = new WebClient();
            // export DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER=0 seems to solve the bugs

            int[] downloadedCounts = new int[sources.Length];

            ProgressBar progressBar = new ProgressBar(domainsByPdb.Keys.Count, $"Downloading {domainsByPdb.Keys.Count} structures ({domains.Length} domains)");

            // TODO nice progress bar
            progressBar.Start();
            foreach (string pdb in domainsByPdb.Keys){
                // string[] urls = sources.Select(source => source.Replace("{pdb}", pdb).Replace("{pdb12}", pdb.Substring(1, 2))).ToArray();
                string[] urls = sources.Select(source => FormatSource(source, pdb)).ToArray();
                string url;
                string cifString;
                try {
                    (url, cifString) = DownloadFirstSuccessful(webClient, urls);
                    downloadedCounts[Array.IndexOf(urls, url)]++;
                } catch (Exception ex) {
                    if (failuresFile == null){
                        throw;
                    } else {
                        string message = pdb;
                        if (failuresFile == "-"){
                            Console.Error.WriteLine(message);
                        } else {
                            using(var r = new StreamWriter(failuresFile, true)){
                                r.WriteLine(message);
                            }
                        }
                        progressBar.Step();
                        continue;
                    }
                }
                CifPackage package = CifPackage.FromString(cifString);
                if (package.Blocks.Length == 0){
                    Console.Error.WriteLine("ERROR: CIF file contains no blocks.");
                    return 3;
                }
                CifBlock block = package.Blocks[0];
                CifCategory atomSiteCategory = block.GetCategory("_atom_site");
                
                if (cifOutDirectory != null || pdbOutDirectory != null){
                    foreach (Domain domain in domainsByPdb[pdb]){
                        Filter filter = Filter.StringEquals("label_asym_id", domain.Chain) & Filter.IntegerInRange("label_seq_id", domain.Ranges);
                        int[] rows = filter.GetFilteredRows(atomSiteCategory).ToArray();

                        // Select only the first model, if there are more than one
                        if (atomSiteCategory.ContainsItem("pdbx_PDB_model_num")){
                            int[] modelNumbers = atomSiteCategory.GetItem("pdbx_PDB_model_num").GetIntegers(rows).Distinct().ToArray();
                            if (modelNumbers.Length == 0) {
                                Console.Error.WriteLine($"Warning: No model found for domain {domain.Name} (or the selection is empty)");
                                break;
                            }
                            if (modelNumbers.Length > 1) {
                                int firstModelNumber = modelNumbers[0];
                                Console.Error.WriteLine($"Warning: More than one model found for domain {domain.Name} (using only model {firstModelNumber})");
                                Filter modelFilter= Filter.IntegerInRange("pdbx_PDB_model_num", (firstModelNumber, firstModelNumber));
                                rows = (Filter.TheseRows(rows) & modelFilter).GetFilteredRows(atomSiteCategory).ToArray();
                            }
                        }

                        if (cifOutDirectory != null){
                            string outputFile = Path.Combine(cifOutDirectory, domain.Name + OUTPUT_EXTENSION);
                            using (StreamWriter w = new StreamWriter(outputFile)){
                                w.WriteLine("#");
                                w.WriteLine("data_" + block.Name);
                                w.WriteLine();
                                if (block.ContainsCategory("_entry")){
                                    w.Write(block.GetCategory("_entry").MakeCifString());
                                }
                                w.Write(atomSiteCategory.MakeCifString(rows));
                            }
                        }

                        if (pdbOutDirectory != null){
                            string pdbOutputFile = Path.Combine(pdbOutDirectory, domain.Name + PDB_OUTPUT_EXTENSION);
                            ModelCollection models = ModelCollection.FromCifBlock(block, rows);
                            if (models.Count == 0) {
                                Console.Error.WriteLine($"Warning: No model found for domain {domain.Name} (or the selection is empty)");
                                break;
                            }
                            Model model = models.GetModel(0);
                            if (models.Count > 1) {
                                Console.Error.WriteLine($"Warning: More than one model found for domain {domain.Name} (using onlymodel {model.ModelNumber})");
                            }
                            RenumberModel(model);
                            try {
                                PrintModelToPdb(model, pdbOutputFile);
                            } catch (PDBFormatSucksException ex) {
                                Console.Error.WriteLine($"ERROR: Cannot fit model for {domain.Name} into PDB format: {ex.Message}");
                                return 4;
                            }
                        }
                    }
                }
                
                if (summaryOutDirectory != null){
                    StructureSummary summary = new StructureSummary(pdb, block);
                    summary.Save(Path.Combine(summaryOutDirectory, pdb + SUMMARY_OUTPUT_EXTENSION));
                }
                progressBar.Step();
            }
            progressBar.Finish();

            for (int i = 0; i < sources.Length; i++){
                Console.WriteLine($"Downloaded {downloadedCounts[i]} PDB entries from {sources[i]}");
            }

            return 0;
        }

        /** Add PDB ID into source, e.g. "http://blabla.org/{pdb_1}{pdb_2}/{pdb}.cif" -> "http://blabla.org/tq/1tqn.cif" */
        static string FormatSource(string source, string pdb){
            source = source.Replace("{pdb}", pdb);
            for (int i = 0; i < 4; i++){
                source = source.Replace($"{{pdb_{i}}}", pdb.Substring(i, 1));
            }
            return source;
        }

        /** Try to decompress a gzipped byte-sequence into a string. If it is not a gzip (does not start with bytes 1f-8b), return null. */
        static string DecompressGzip(byte[] gzipped){
            bool isGzip = gzipped[0] == '\x1f' && gzipped[1] == '\x8b';
            if (!isGzip){
                return null;
            } else {
                string result;;
                using (Stream origStream = new MemoryStream(gzipped)){
                    using (GZipStream gzStream = new GZipStream(origStream, CompressionMode.Decompress)){
                        using (StreamReader reader = new StreamReader(gzStream)){
                            result = reader.ReadToEnd();
                        }
                    }
                }
                return result;
            }
        }

        static string DownloadString(string url){
            WebClient client = new WebClient();
            string content = client.DownloadString(url);
            return content;
        }

        static (string url, string content) DownloadFirstSuccessful(WebClient webClient, string[] urls){
            Exception lastException = null;
            foreach(string url in urls){
                if (!Uri.IsWellFormedUriString(url, UriKind.RelativeOrAbsolute)){
                    throw new ArgumentException($"Invalid URI: {url}");
                }
                try {
                    // string content = webClient.DownloadString(url);
                    byte[] bytes = webClient.DownloadData(url);
                    string content = DecompressGzip(bytes) ?? webClient.Encoding.GetString(bytes);
                    return (url, content);
                } catch (WebException ex){
                    lastException = ex;
                }
            }
            throw new Exception($"Could not get any of these URLs: {String.Join(' ', urls)}", lastException);
        }

        ///<summary> Changes chain and residue numbering in model (chainID = DEFAULT_CHAIN_ID, resi = numbered sequentially starting from DEFAULT_CHAIN_ID). </summary>
        private static void RenumberModel_KeepAtomIds(Model model) {
            if (model.Chains.Count != 1){
                throw new NotImplementedException("Not implemented for more than 1 chain in domain!");
            } 
            model.Chains.Id[0] = DEFAULT_CHAIN_ID.ToString();
            int[] resis = model.Residues.SeqNumber;
            int resi = 1;
            for (int i = 0; i < model.Residues.Count-1; i++) {
                int oldResi = resis[i];
                resis[i] = resi;
                    resi++;
                if (resis[i+1] > oldResi+1){ // missing residues in the chain => skip residue numbers
                    resi += SKIPPED_RESIDUES_AT_GAP;
                }
            }
            resis[model.Residues.Count-1] = resi;
        }

        ///<summary> Changes chain, residue, and atom numbering in model (chainID = DEFAULT_CHAIN_ID, resi and atomID = numbered sequentially starting from 1). </summary>
        private static void RenumberModel(Model model) {
            if (model.Chains.Count != 1){
                throw new NotImplementedException("Not implemented for more than 1 chain in domain!");
            } 
            model.Chains.Id[0] = DEFAULT_CHAIN_ID.ToString();

            int[] resis = model.Residues.SeqNumber;
            int resi = 1;
            for (int i = 0; i < model.Residues.Count-1; i++) {
                int oldResi = resis[i];
                resis[i] = resi;
                    resi++;
                if (resis[i+1] > oldResi+1){ // missing residues in the chain => skip residue numbers
                    resi += SKIPPED_RESIDUES_AT_GAP;
                }
            }
            resis[model.Residues.Count-1] = resi;

            for (int i = 0; i < model.Atoms.Count; i++) {
                model.Atoms.Id[i] = (i + 1).ToString();
            }
        }

        private static void PrintModelToPdb(Model model, string outputFile){
            AtomTable atoms = model.Atoms;
            ResidueTable residues = model.Residues;
            ChainTable chains = model.Chains;
            const char INSERTION_CODE = ' ';
            const string CHARGE = "  ";
            const string OCCUPANCY = "   .  "; // "  1.00";
            const string TEMP_FACTOR = "   .  "; // "  0.00";

            using (StreamWriter w = new StreamWriter(outputFile)){
                for (int ai = 0; ai < atoms.Count; ai++){

                    string atomString = (atoms.IsHetatm[ai] ? "HETATM" : "ATOM  ")
                        + SafePadLeft(atoms.Id[ai].ToString(), 5, "Atom ID")
                        + " "
                        + PdbStyleAtomName(atoms.Name[ai], atoms.Element[ai])
                        + SafePadLeft(atoms.AltLoc[ai]=="." ? " " : atoms.AltLoc[ai], 1, "Alternate location indicator")
                        + SafePadLeft(residues.Compound[atoms.ResidueIndex[ai]], 3, "Residue name")
                        + " "
                        + SafePadLeft(chains.Id[atoms.ChainIndex[ai]], 1, "Chain ID")
                        + SafePadLeft(residues.SeqNumber[atoms.ResidueIndex[ai]].ToString(), 4, "Residue number")
                        + INSERTION_CODE
                        + "   "
                        + SafePadLeft(atoms.X[ai].ToString("0.000"/*, new CultureInfo("en-US")*/), 8, "X coordinate")
                        + SafePadLeft(atoms.Y[ai].ToString("0.000"/*, new CultureInfo("en-US")*/), 8, "Y coordinate")
                        + SafePadLeft(atoms.Z[ai].ToString("0.000"/*, new CultureInfo("en-US")*/), 8, "Z coordinate")
                        + OCCUPANCY
                        + TEMP_FACTOR
                        + "          "
                        + SafePadLeft(atoms.Element[ai], 2, "Element symbol")
                        + CHARGE;
                    // Console.WriteLine(atomString);
                    w.WriteLine(atomString);
                }
            }
        }

        private static string SafePadLeft(string str, int width, string fieldName = "Value"){
            if (str.Length <= width) {
                return str.PadLeft(width);
            } else {
                throw new PDBFormatSucksException(fieldName, width, str);
            }
        }

        private static string SafePadRight(string str, int width, string fieldName = "Value"){
            if (str.Length <= width) {
                return str.PadRight(width);
            } else {
                throw new PDBFormatSucksException(fieldName, width, str);
            }
        }

        private static string PdbStyleAtomName(string atomName, string element){
            atomName = atomName.Trim();
            element = element.Trim();
            if (element.Length == 1 && atomName.Length <= 3){
                return SafePadRight(" " + atomName, 4, "Atom name");
            } else {
                return SafePadRight(atomName, 4, "Atom name");
            }
        }

    }
}
