﻿using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Linq;
using System.Json;  // installed with: dotnet add package System.Json --version 4.5.0
using Cif;
using Cif.Filtering;
using Cif.Tables;

namespace StructureCutter
{
    public class Program
    {   
        const string DEFAULT_URL_SOURCE = "http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif";
        // const string DEFAULT_URL_SOURCE = "file:///home/adam/Workspace/Python/Ubertemplate/data/GuidedAcyclicClustering/cif_cache/{pdb}.cif";
        static readonly string[] DEFAULT_URL_SOURCES = { "http://www.ebi.ac.uk/pdbe/entry-files/download/{pdb}_updated.cif" };
        const string OUTPUT_EXTENSION = ".cif";
        const string PDB_OUTPUT_EXTENSION = ".pdb";
        const char DEFAULT_CHAIN_ID = '.';  // Assigned to chains with multi-character IDs, when converting to PDB.
        const int FIRST_RESIDUE_ID = 1;  // Assigned to the first residue, when converting to PDB.
        const int SKIPPED_RESIDUES_AT_GAP = 1;  // Number of residue numbers to be skipped when some residues are missing in a chain, when converting to PDB.

        static int Main(string[] args)
        {
            string cifOutDirectory = "";
            string pdbOutDirectory = "";
            string[] sources = DEFAULT_URL_SOURCES;
            string failuresFile = null;

            Options options = new Options();
            options.GlobalHelp = "StructureCutter 0.9";
            options.AddArgument(new Argument("DOMAIN_LIST_FILE").AddHelp("JSON file with the list of domains to be downloaded, in format [[pdb, domain_name, chain, ranges]*]"));

            options.AddOption(Option.StringOption(new string[]{"--cif_outdir"}, v => { cifOutDirectory = v; })
                .AddParameter("DIRECTORY")
                .AddHelp("Directory for output files in mmCIF format (or empty string to suppress mmCIF output).")
            );
            options.AddOption(Option.StringOption(new string[]{"--pdb_outdir"}, v => { pdbOutDirectory = v; })
                .AddParameter("DIRECTORY")
                .AddHelp("Directory for output files in PDB format (or empty string to suppress PDB output).")
            );
            options.AddOption(Option.StringOption(new string[]{"--sources"}, v => { sources = v.Split(' ', StringSplitOptions.RemoveEmptyEntries); })
                .AddParameter("SOURCES")
                .AddHelp("Space-separated list of URL sources (each starting with http:// or file:///).")
                .AddHelp("{pdb} in each source will be replaced by the actual PDB ID, {pdb12} will be replaced by the 2 middle characters of the PDB ID.")
                .AddHelp($"Default: '{String.Join(' ', DEFAULT_URL_SOURCES)}'")
            );
            options.AddOption(Option.StringOption(new string[]{"--failures"}, v => { failuresFile = v; })
                .AddParameter("FILE")
                .AddHelp("Filename for output of failed PDBIDs. Use '-' to output on stderr. Default is to exit on any failures.")
            );

            List<string> otherArgs;
            bool optionsOK = options.TryParse(args, out otherArgs);
            if (!optionsOK){
                return 1;
            }
            if (otherArgs.Count != 1){
                Options.PrintError($"Wrong number arguments (expected 1, got {otherArgs.Count}).");
                return 1;
            }
            string domainListFile = otherArgs[0];


            Console.WriteLine($"Domain list file: {domainListFile}");
            Console.WriteLine($"Sources:");
            foreach(string source in sources){
                Console.WriteLine($"\t{source}");
            }
            Console.WriteLine($"CIF output directory: {cifOutDirectory}");
            Console.WriteLine($"PDB output directory: {pdbOutDirectory}");

            if (sources.Length == 0){
                Options.PrintError($"Must specify at least one source.");
                return 1;
            }
            if (cifOutDirectory == "" && pdbOutDirectory == ""){
                Console.WriteLine("WARNING: You did not specify either of --cif_outdir, --pdb_outdir. No output will be produced.");
            }
            
            // // Parse arguments
            // if (args.Length != 3){
            //     Console.Error.WriteLine("ERROR: Wrong number arguments.");
            //     Console.Error.WriteLine($"  Usage: {AppDomain.CurrentDomain.FriendlyName} DOMAIN_LIST_FILE OUTPUT_DIRECTORY PDB_OUTPUT_DIRECTORY");    
            //     return 1;       
            // }
            // string domainListFile = args[0];
            // string cifOutDirectory = args[1];
            // string pdbOutDirectory = args[2];

            // TODO write documentation

            Domain[] domains = ReadDomainsFromJson(domainListFile);
            Dictionary<string,List<Domain>> domainsByPdb = SortDomainsByPdb(domains);

            if (cifOutDirectory != ""){
                Directory.CreateDirectory(cifOutDirectory);
            }
            if (pdbOutDirectory != ""){
                Directory.CreateDirectory(pdbOutDirectory);
            }
            if (failuresFile != null && failuresFile != "-"){
                using(var r = new StreamWriter(failuresFile, false)){
                    // clear file
                }
            }

            WebClient webClient = new WebClient();

            int[] downloadedCounts = new int[sources.Length];

            ProgressBar progressBar = new ProgressBar(domainsByPdb.Keys.Count, $"Downloading {domainsByPdb.Keys.Count} structures ({domains.Length} domains)");

            // TODO nice progress bar
            progressBar.Start();
            foreach (string pdb in domainsByPdb.Keys){
                string[] urls = sources.Select(source => source.Replace("{pdb}", pdb).Replace("{pdb12}", pdb.Substring(1, 2))).ToArray();
                string url;
                string cifString;
                try {
                    (url, cifString) = DownloadFirstSuccessful(webClient, urls);
                    downloadedCounts[Array.IndexOf(urls, url)]++;
                } catch (Exception ex) {
                    if (failuresFile == null){
                        throw ex;
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
                    Console.Error.WriteLine("CIF file contains no blocks.");
                    return 1;
                }
                CifBlock block = package.Blocks[0];
                CifCategory atomSiteCategory = block.GetCategory("_atom_site");

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

                    if (cifOutDirectory != ""){
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

                    if (pdbOutDirectory != ""){
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
                        PrintModelToPdb(model, pdbOutputFile);
                    }
                }
                progressBar.Step();
            }
            progressBar.Finish();

            for (int i = 0; i < sources.Length; i++){
                Console.WriteLine($"Downloaded {downloadedCounts[i]} PDB entries from {sources[i]}");
            }

            // Stream stream = new FileStream("el_file.gz", FileMode.Open);
            // GZipStream ungzipper = new GZipStream(stream, CompressionMode.Decompress);

            return 0;
        }

        // static void ProcessPDB(string cifString){
        //     CifPackage package = CifPackage.FromString(cifString);
        //     if (package.Blocks.Length == 0){
        //         throw new Exception("CIF file contains no blocks.");
        //     }
        //     CifBlock block = package.Blocks[0];
        //     CifCategory atomSiteCategory = block.GetCategory("_atom_site");

        //     foreach (Domain domain in domainsByPdb[pdb]){
        //         Filter filter = Filter.StringEquals("label_asym_id", domain.Chain) & Filter.IntegerInRange("label_seq_id", domain.Ranges);
        //         int[] rows = filter.GetFilteredRows(atomSiteCategory).ToArray();

        //         // Select only the first model, if there are more than one
        //         if (atomSiteCategory.ContainsItem("pdbx_PDB_model_num")){
        //             int[] modelNumbers = atomSiteCategory.GetItem("pdbx_PDB_model_num").GetIntegers(rows).Distinct().ToArray();
        //             if (modelNumbers.Length == 0) {
        //                 Console.Error.WriteLine($"Warning: No model found for domain {domain.Name} (or the selection is empty)");
        //                 break;
        //             }
        //             if (modelNumbers.Length > 1) {
        //                 int firstModelNumber = modelNumbers[0];
        //                 Console.Error.WriteLine($"Warning: More than one model found for domain {domain.Name} (using only model {firstModelNumber})");
        //                 Filter modelFilter= Filter.IntegerInRange("pdbx_PDB_model_num", (firstModelNumber, firstModelNumber));
        //                 rows = (Filter.TheseRows(rows) & modelFilter).GetFilteredRows(atomSiteCategory).ToArray();
        //             }
        //         }

        //         if (cifOutDirectory != ""){
        //             string outputFile = Path.Combine(cifOutDirectory, domain.Name + OUTPUT_EXTENSION);
        //             using (StreamWriter w = new StreamWriter(outputFile)){
        //                 w.WriteLine("#");
        //                 w.WriteLine("data_" + block.Name);
        //                 w.WriteLine();
        //                 if (block.ContainsCategory("_entry")){
        //                     w.Write(block.GetCategory("_entry").MakeCifString());
        //                 }
        //                 w.Write(atomSiteCategory.MakeCifString(rows));
        //             }
        //         }

        //         if (pdbOutDirectory != ""){
        //             string pdbOutputFile = Path.Combine(pdbOutDirectory, domain.Name + PDB_OUTPUT_EXTENSION);
        //             ModelCollection models = ModelCollection.FromCifBlock(block, rows);
        //             if (models.Count == 0) {
        //                 Console.Error.WriteLine($"Warning: No model found for domain {domain.Name} (or the selection is empty)");
        //                 break;
        //             }
        //             Model model = models.GetModel(0);
        //             if (models.Count > 1) {
        //                 Console.Error.WriteLine($"Warning: More than one model found for domain {domain.Name} (using onlymodel {model.ModelNumber})");
        //             }
        //             RenumberModel(model);
        //             PrintModelToPdb(model, pdbOutputFile);
        //         }
        //     }
        // }

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

        private static Domain[] ReadDomainsFromJson(string filename){
            string domainsJsonString;
            using (StreamReader r = new StreamReader(filename)){
                domainsJsonString = r.ReadToEnd();
            }
            JsonValue domainsJson = System.Json.JsonValue.Parse(domainsJsonString);
            if (domainsJson.JsonType != JsonType.Array) {
                throw new FormatException($"Content of {filename} is not JSON array.");
            }
            int nDomains = domainsJson.Count;
            Domain[] domains = new Domain[nDomains];
            for (int i = 0; i<nDomains; i++){
                JsonValue domainJson = domainsJson[i];
                if (domainJson.JsonType != JsonType.Array) {
                    throw new FormatException($"Content of {filename}[{i}] is not JSON array.");
                }
                if (domainJson.Count != 4) {
                    throw new FormatException($"Content of {filename}[{i}] has length different from 4.");
                }
                if ((domainJson as IEnumerable<JsonValue>).Any(val => val.JsonType != JsonType.String)) {
                    throw new FormatException($"Content of {filename}[{i}] has element which are not string.");
                }
                string pdb = domainJson[0];
                string domainName = domainJson[1];
                string chain = domainJson[2];
                string range = domainJson[3];
                domains[i] = new Domain(domainName, pdb, chain, range);
            }
            return domains;
        }

        private static Dictionary<string,List<Domain>> SortDomainsByPdb(IEnumerable<Domain> domains){
            Dictionary<string,List<Domain>> result = new Dictionary<string, List<Domain>>();
            foreach (Domain domain in domains){
                if (!result.ContainsKey(domain.Pdb)){
                    result[domain.Pdb] = new List<Domain>();
                }
                result[domain.Pdb].Add(domain);
            }
            return result;
        }

        ///<summary> Changes chain and residue numbering in model (chainID = DEFAULT_CHAIN_ID, resi = numbered sequentially starting from DEFAULT_CHAIN_ID). </summary>
        private static void RenumberModel(Model model) {
            if (model.Chains.Count != 1){
                throw new NotImplementedException("Not implemented for more than 1 chain in domain!");
            } 
            // if (model.Chains.Id[0].Length != 1){
            //     model.Chains.Id[0] = DEFAULT_CHAIN_ID.ToString();
            // }
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
                        + SafePadLeft(atoms.Id[ai].ToString(), 5)
                        + " "
                        + PdbStyleAtomName(atoms.Name[ai], atoms.Element[ai])
                        + SafePadLeft(atoms.AltLoc[ai]=="." ? " " : atoms.AltLoc[ai], 1)
                        + SafePadLeft(residues.Compound[atoms.ResidueIndex[ai]], 3)
                        + " "
                        + SafePadLeft(chains.Id[atoms.ChainIndex[ai]], 1)
                        + SafePadLeft(residues.SeqNumber[atoms.ResidueIndex[ai]].ToString(), 4)
                        + INSERTION_CODE
                        + "   "
                        + SafePadLeft(atoms.X[ai].ToString("0.000"/*, new CultureInfo("en-US")*/), 8)
                        + SafePadLeft(atoms.Y[ai].ToString("0.000"/*, new CultureInfo("en-US")*/), 8)
                        + SafePadLeft(atoms.Z[ai].ToString("0.000"/*, new CultureInfo("en-US")*/), 8)
                        + OCCUPANCY
                        + TEMP_FACTOR
                        + "          "
                        + SafePadLeft(atoms.Element[ai], 2)
                        + CHARGE;
                    // Console.WriteLine(atomString);
                    w.WriteLine(atomString);
                }
            }
        }

        private static string SafePadLeft(string str, int width){
            if (str.Length <= width) {
                return str.PadLeft(width);
            } else {
                throw new Exception($"Could not fit string '{str}' into {width} characters.");
            }
        }

        private static string SafePadRight(string str, int width){
            if (str.Length <= width) {
                return str.PadRight(width);
            } else {
                throw new Exception($"Could not fit string '{str}' into {width} characters.");
            }
        }

        private static string PdbStyleAtomName(string atomName, string element){
            atomName = atomName.Trim();
            element = element.Trim();
            if (element.Length == 1 && atomName.Length <= 3){
                return SafePadRight(" " + atomName, 4);
            } else {
                return SafePadRight(atomName, 4);
            }
        }

    }
}
