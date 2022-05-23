using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace StructureCutter
{
    public static class Lib 
    {
        public static void PrintError(string message)
        {
            WriteInColor(ConsoleColor.Red, message, file: Console.Error);
        }

        public static void WriteInColor(ConsoleColor color, string text, TextWriter file = null, bool newline = true)
        {
            file = file ?? Console.Out;
            ConsoleColor orig = Console.ForegroundColor;
            Console.ForegroundColor = color;
            file.Write(text);
            if (newline) file.WriteLine();
            Console.ForegroundColor = orig;
        }

        public static void Replace<T>(this T[] array, T oldValue, T newValue)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i].Equals(oldValue)){
                    array[i] = newValue;
                }
            }
        }
    }
}