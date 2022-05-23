using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Json;  // installed with: dotnet add package System.Json --version 4.5.0

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
            if (rangeString == null) return null;
            return rangeString.Split(',').Select(range => ParseOneRangeString(range)).ToArray();
        }

        
        public static Domain[] ReadDomainsFromJson(string filename){
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
                try {
                    domains[i] = FromJson(domainsJson[i]);
                } catch (FormatException ex){
                    throw new FormatException($"Content of {filename}[{i}] cannot be parsed as a domain. {ex.Message}");
                }
            }
            return domains;
        }
    
        private static Domain FromJson(JsonValue domainJson){
            string pdb;
            string domainName;
            string chain;
            string range;
            if (domainJson.JsonType == JsonType.Array) {
                if (domainJson.Count == 4) {
                    pdb = JsonValueToString(domainJson[0], valueName: "Content[0]", mustBe: "a string (PDB ID)");
                    domainName = JsonValueToString(domainJson[1], valueName: "Content[1]", mustBe: "a string (domain name)");
                    chain = JsonValueToString(domainJson[2], valueName: "Content[2]", mustBe: "a string (chain ID)");
                    range = JsonValueToString(domainJson[3], valueName: "Content[3]", mustBe: "a string (residue ranges)");
                } else {       
                    throw new FormatException($"Content must contain 4 elements (PDB ID, domain name, chain ID, residue ranges).");
                }
            } else if (domainJson.JsonType == JsonType.Object) {
                try{
                    JsonObject domainObj = domainJson as JsonObject;
                    domainName = JsonValueToString(GetJsonObjectValue(domainObj, "domain"), valueName: "Content[\"domain\"]", mustBe: "a string (domain name)");
                    pdb = JsonValueToString(GetJsonObjectValue(domainObj, "pdb"), valueName: "Content[\"pdb\"]", mustBe: "a string (PDB ID)");
                    chain = JsonValueToString(GetJsonObjectValue(domainObj, "chain_id"), valueName: "Content[\"chain_id\"]", mustBe: "a string (chain ID)", allowNull: true);
                    range = JsonValueToString(GetJsonObjectValue(domainObj, "ranges"), valueName: "Content[\"ranges\"]", mustBe: "a string (residue ranges)", allowNull: true);
                    if (chain == "") chain = null;
                    if (range == "") range = null;
                } catch (KeyNotFoundException ex) {
                    string missingKey = ex.Message;
                    throw new FormatException($"Content must contain key \"{missingKey}\".");
                }
            } else {
                throw new FormatException($"Content must be a JSON object or a JSON array.");
            }
            return new Domain(domainName, pdb, chain, range);
        }

        private static JsonValue GetJsonObjectValue(JsonObject obj, string key){
            try {
                return obj[key];
            } catch (KeyNotFoundException) {
                throw new KeyNotFoundException(key);
            }
        }

        private static string JsonValueToString(JsonValue jsonValue, string valueName = "Content", string mustBe = "a string", bool allowNull = false){
            if (jsonValue == null){
                if (allowNull) 
                    return null;
                else
                    throw new FormatException($"{valueName} must be {mustBe}, not null.");
            }
            if (jsonValue.JsonType != JsonType.String){
                throw new FormatException($"{valueName} must be {mustBe}, not {jsonValue}.");
            }
            return jsonValue;
        }


        public static Dictionary<string,List<Domain>> SortDomainsByPdb(IEnumerable<Domain> domains){
            Dictionary<string,List<Domain>> result = new Dictionary<string, List<Domain>>();
            foreach (Domain domain in domains){
                if (!result.ContainsKey(domain.Pdb)){
                    result[domain.Pdb] = new List<Domain>();
                }
                result[domain.Pdb].Add(domain);
            }
            return result;
        }

    }
}