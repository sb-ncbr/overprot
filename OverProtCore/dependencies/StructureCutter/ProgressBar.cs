using System;
using System.IO;

namespace StructureCutter
{
    public class ProgressBar
    {
        public const int DEFAULT_WIDTH = 100;
        public static readonly TextWriter DEFAULT_WRITER = Console.Out;

        public int TotalSteps { get; private set; }
        public int Width { get; private set; }
        public string Title { get; private set; }
        public TextWriter Writer { get; private set; }
        public int StepsDone { get; private set; }
        public int SymbolsShown { get; private set; }

        public ProgressBar(int nSteps, string title) : this(nSteps, title, DEFAULT_WIDTH, DEFAULT_WRITER) {}
        public ProgressBar(int nSteps, string title, int width, TextWriter writer){
            TotalSteps = nSteps; // expected number of steps
            Width = width;
            Title = (" " + title + " ").Substring(0, Math.Min(title.Length+2, Width));
            Writer = writer;
            StepsDone = 0; // number of completed steps
            SymbolsShown = 0; // number of shown symbols
        }

        public void Start(){
            Writer.WriteLine("|" + Title.PadRight(Width, '_') + "|");
            Writer.Write("|");
            Step(0);
        }

        public void Step() => Step(1);
        public void Step(int nSteps){
            StepsDone = Math.Min(StepsDone + nSteps, TotalSteps);
            int newShown = TotalSteps>0 ? Convert.ToInt32(Math.Floor(1.0 * Width * StepsDone / TotalSteps)) : Width;
            for (; SymbolsShown < newShown; SymbolsShown++) {
                Writer.Write("*");
            }
        }

        public void Finish(){
            Step(TotalSteps - StepsDone);
            Writer.WriteLine("|");
        }

    }
}