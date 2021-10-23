using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Json;  // installed with: dotnet add package System.Json --version 4.5.0

namespace StructureCutter
{
    public class PDBFormatSucksException : Exception
    {
        public readonly string FieldName;
        public readonly int MaxWidth;
        public readonly string FieldValue;
        public override string Message => $"{FieldName} '{FieldValue}' cannot fit into {MaxWidth} characters.";

        public PDBFormatSucksException(string fieldName, int maxWidth, string fieldValue) : base() {
            this.FieldName = fieldName;
            this.MaxWidth = maxWidth;
            this.FieldValue = fieldValue;
        }
    }
}