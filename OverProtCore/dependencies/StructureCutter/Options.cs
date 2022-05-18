using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace StructureCutter
{
    public class OptionParseException : Exception
    {
        public OptionParseException(String message) : base(message) { }
        public OptionParseException(String message, Exception innerException) : base(message, innerException) { }
    }

    public class Null { }

    public class Box<K>
    {
        public K Value { get; private set; }
        public Box(K value) { Value = value; }
        public override string ToString() => $"Box[{Value}]";
    }


    public class Options
    {
        public String GlobalHelp { get; set; }
        private List<IOption<object>> ArgumentList { get; }
        private List<IOption<object>> OptionList { get; }
        private Dictionary<string, IOption<object>> OptionDict { get; }
        private const String OPTION_INDENT = "  ";
        private const String OPTION_HELP_INDENT = "          ";

        public Options()
        {
            this.ArgumentList = new List<IOption<object>>();
            this.OptionList = new List<IOption<object>>();
            this.OptionDict = new Dictionary<string, IOption<object>>();
            this.AddHelpOption("-h", "--help").AddHelp("Print this help message and exit.");
        }

        public Option<T> AddOption<T>(Option<T> option)
        {
            foreach (string name in option.Names)
            {
                if (OptionDict.ContainsKey(name))
                {
                    throw new ArgumentException($"Option conflict: multiple definitions for option {name}");
                }
                OptionDict[name] = (IOption<object>) option;
            }
            if (option.IsOption)
                this.OptionList.Add((IOption<object>) option);
            else
                this.ArgumentList.Add((IOption<object>) option);
            return option;
        }

        public Option<T> AddOption<T>(int numArgs, Func<string[], T> parser, params string[] names)
        {
            Option<T> option = new Option<T>(numArgs, parser, names);
            this.AddOption(option);
            return option;
        }

        public SwitchOption AddSwitchOption(params string[] names)
        {
            SwitchOption option = new SwitchOption(names);
            this.AddOption(option);
            return option;
        }
        public IntOption AddIntOption(params string[] names)
        {
            IntOption option = new IntOption(names);
            this.AddOption(option);
            return option;
        }
        public DoubleOption AddDoubleOption(params string[] names)
        {
            DoubleOption option = new DoubleOption(names);
            this.AddOption(option);
            return option;
        }
        public StringOption AddStringOption(params string[] names)
        {
            StringOption option = new StringOption(names);
            this.AddOption(option);
            return option;
        }
        public ChoiceOption AddChoiceOption(IEnumerable<string> choices, params string[] names)
        {
            ChoiceOption option = new ChoiceOption(choices, names);
            this.AddOption(option);
            return option;
        }
        public DictionaryChoiceOption<T> AddDictionaryChoiceOption<T>(Dictionary<T, string> dictionary, params string[] names)
        {
            DictionaryChoiceOption<T> option = new DictionaryChoiceOption<T>(dictionary, names);
            this.AddOption(option);
            return option;
        }
        public HelpOption AddHelpOption(params string[] names)
        {
            HelpOption option = new HelpOption(this, names);
            this.AddOption(option);
            return option;
        }


        private void ParseOption(IOption<object> option, string optionName, Queue<string> argQueue){
            if (argQueue.Count < option.NumArgs)
            {
                string message = option.NumArgs==1 ? "Missing value." : $"Requires {option.NumArgs} values, got {argQueue.Count}.";
                throw new OptionParseException(message);
            }
            string[] optionArgs = argQueue.Pop(option.NumArgs);
            foreach (var cm in option.Constraints)
            {
                if (!cm.constraint(optionArgs))
                {
                    throw new OptionParseException("\"" + string.Join(" ", optionArgs) + "\", " + cm.message);
                }
            }
            option.SetInternalValue(optionArgs);
            foreach (var action in option.Actions)
            {
                action();
            }
        }
        
        public bool TryParse(IEnumerable<String> args){
            string[] extraArgs;
            bool ok = TryParse(args, out extraArgs);
            if (!ok) return false;
            if (extraArgs.Length > 0){
                PrintError($"Too many arguments: " + string.Join(' ', extraArgs));
                return false;
            }
            return true;
        }
        public bool TryParse(IEnumerable<String> args, out string[] extraArgs)
        {
            extraArgs = null;
            Queue<string> argQueue = new Queue<string>(args);
            List<string> remainingArgs = new List<String>();
            while (argQueue.Count > 0)
            {
                string name = argQueue.Pop();
                if (name.Length > 0 && name[0] == '-')
                {
                    if (OptionDict.ContainsKey(name)){
                        IOption<object> option = OptionDict[name];
                        try {
                            ParseOption(option, name, argQueue);
                        } catch (OptionParseException e) {
                            PrintError($"Option {name}: {e.Message}");
                            return false;
                        }
                    } else {
                        PrintError($"Unknown option: {name}.");
                        return false;
                    }
                }
                else
                {
                    remainingArgs.Add(name);
                }
            }
            argQueue = new Queue<string>(remainingArgs);
            foreach (var option in ArgumentList){
                try {
                    ParseOption(option, option.Names[0], argQueue);
                } catch (OptionParseException e) {
                    PrintError($"Argument {option.Names[0]}: {e.Message}");
                    return false;
                }
            }
            extraArgs = argQueue.Pop(argQueue.Count);
            return true;
        }

        public void PrintHelp()
        {
            Console.WriteLine(GlobalHelp);

            Console.WriteLine("\nUsage:");
            Console.WriteLine(OPTION_INDENT + "dotnet " + System.AppDomain.CurrentDomain.FriendlyName + ".dll [options] " + string.Join(" ", ArgumentList.Select(a => a.Names[0])));
            Console.WriteLine("\nArguments:");
            foreach (IOption<object> argument in ArgumentList)
            {
                String line = OPTION_INDENT + argument.Names[0];
                WriteInColor(ConsoleColor.Cyan, line);
                foreach (String help in argument.Helps)
                {
                    Console.WriteLine(OPTION_HELP_INDENT + help);
                }
            }

            Console.WriteLine("\nOptions:");
            foreach (IOption<object> option in OptionList)
            {
                String line = OPTION_INDENT + string.Join(", ", option.Names) + " " + string.Join(" ", option.Parameters);
                WriteInColor(ConsoleColor.Cyan, line);
                foreach (String help in option.Helps)
                {
                    Console.WriteLine(OPTION_HELP_INDENT + help);
                }
            }
            Environment.Exit(0);
        }

        public static void PrintError(string message)
        {
            string appName = System.AppDomain.CurrentDomain.FriendlyName;
            WriteInColor(ConsoleColor.Red, message, file: Console.Error);
            WriteInColor(ConsoleColor.Red, $"Run   dotnet {appName}.dll --help   for usage information.", file: Console.Error);
        }

        private static void WriteInColor(ConsoleColor color, string text, TextWriter file = null, bool newline = true)
        {
            file = file ?? Console.Out;
            ConsoleColor orig = Console.ForegroundColor;
            Console.ForegroundColor = color;
            file.Write(text);
            if (newline) file.WriteLine();
            Console.ForegroundColor = orig;
        }

    }


    public interface IOption<out T>
    {
        String[] Names { get; }
        int NumArgs { get; }
        Func<string[], T> Parser { get; }
        List<Action> Actions { get; }
        List<(Func<string[], bool> constraint, string message)> Constraints { get; }
        List<String> Parameters { get; }
        List<String> Helps { get; }
        public void SetInternalValue(string[] args);
    }

    public class Option<T> : IOption<T>
    {
        public string[] Names { get; private set; }
        public bool IsOption { get; private set; }
        public int NumArgs { get; private set; }
        public Func<string[], T> Parser { get; private set; }
        public List<Action> Actions { get; private set; }
        public List<(Func<string[], bool>, string)> Constraints { get; private set; }
        public List<string> Parameters { get; private set; }
        public List<string> Helps { get; private set; }
        protected T InternalValue { get; set; }
        public T Value { get => InternalValue; }
        public void SetInternalValue(string[] args)
        {
            InternalValue = Parser(args);
        }

        public Option(int numArgs, Func<string[], T> parser, params string[] names)
        {
            if (names.Length == 0)
                throw new ArgumentException("There must be at least one argument/option name.");
            if (names.Any(name => name == null || name.Length == 0))
                throw new ArgumentException("Argument/option names must not be empty or null.");
            if (names[0][0] == '-') {
                IsOption = true;
                if (names.Any(name => name[0] != '-'))
                    throw new ArgumentException("Option names must begin with -.");
            } else {
                IsOption = false;
                if (names.Length > 1)
                    throw new ArgumentException("Positional argument can only have one name.");
            }
            Names = names;
            NumArgs = numArgs;
            Parser = parser;
            Actions = new List<Action>();
            Constraints = new List<(Func<string[], bool>, string)>();
            Parameters = new List<string>();
            Helps = new List<string>();
        }

        public Option<T> AddAction(Action action)
        {
            this.Actions.Add(action);
            return this;
        }

        public Option<T> AddConstraint(Func<string[], bool> constraint, string errorMessage)
        {
            Constraints.Add((constraint, errorMessage));
            return this;
        }

        public Option<T> AddParameter(string parameter)
        {
            Parameters.Add(parameter);
            return this;
        }

        public Option<T> AddHelp(string helpMessage)
        {
            Helps.Add(helpMessage);
            return this;
        }

        public Option<T> SetDefault(T value)
        {
            InternalValue = value;
            return this;
        }

        public override string ToString() => $"Argument/option {Names[0]} [{InternalValue}]";

    }


    public class SwitchOption : Option<Box<bool>>
    {
        public new bool Value { get => InternalValue.Value; }
        public SwitchOption(params string[] names) : base(0, args => new Box<bool>(true), names)
        {
            this.InternalValue = new Box<bool>(false);
        }
    }

    public class IntOption : Option<Box<int>>
    {
        public new int Value { get => InternalValue.Value; }
        public IntOption(params string[] names) : base(1, args => new Box<int>(ParseInt(args[0])), names)
        {
            this.InternalValue = new Box<int>(0);
        }
        public IntOption SetDefault(int value){
            InternalValue = new Box<int>(value);
            return this;
        }
        private static int ParseInt(string str)
        {
            try
            {
                int i = Int32.Parse(str);
                return i;
            }
            catch
            {
                throw new OptionParseException($"Could not parse \"{str}\" as integer.");
            }
        }
    }

    public class DoubleOption : Option<Box<double>>
    {
        public new double Value { get => InternalValue.Value; }
        public DoubleOption(params string[] names) : base(1, args => new Box<double>(ParseDouble(args[0])), names)
        {
            this.InternalValue = new Box<double>(0.0);
        }
        public DoubleOption SetDefault(double value){
            InternalValue = new Box<double>(value);
            return this;
        }
        private static double ParseDouble(string str)
        {
            try
            {
                double i = Double.Parse(str);
                return i;
            }
            catch
            {
                throw new OptionParseException($"Could not parse \"{str}\" as float.");
            }
        }

    }

    public class StringOption : Option<string>
    {
        public StringOption(params string[] names) : base(1, args => args[0], names) { }
        public new StringOption SetDefault(string value){
            InternalValue = value;
            return this;
        }
    }

    public class ChoiceOption : Option<string>
    {
        private IEnumerable<string> choices;
        public ChoiceOption(IEnumerable<string> choices, params string[] names) : base(1, args => ParseChoice(args[0], choices), names) { 
            this.choices = choices;
            if (choices.Count() == 0) throw new ArgumentException($"Argument/option {names[0]}: choices must contain at least one value.");
            InternalValue = choices.First();
        }
        public new ChoiceOption SetDefault(string value){
            InternalValue = ParseChoice(value, choices);
            return this;
        }
        private static string ParseChoice(string str, IEnumerable<String> choices)
        {
            if (choices.Contains(str))
            {
                return str;
            }
            else
            {
                throw new OptionParseException($"Invalid choice \"{str}\" (valid choices: " + string.Join(", ", choices) + ").");
            }
        }
    }

    public class DictionaryChoiceOption<K> : Option<Box<K>>
    {
        private Dictionary<K, String> dictionary;
        public new K Value { get => InternalValue.Value; }
        public DictionaryChoiceOption(Dictionary<K, String> dictionary, params string[] names) : base(1, args => new Box<K>(ParseDictionaryChoice(args[0], dictionary)), names) {
            this.dictionary = dictionary;
            if (dictionary.Count == 0) throw new ArgumentException($"Argument/option {names[0]}: dictionary must have at least one key-value pair.");
            InternalValue = new Box<K>(dictionary.First().Key);
        }
        public DictionaryChoiceOption<K> SetDefault(K value){
            if (!dictionary.ContainsKey(value)) throw new OptionParseException($"Invalid choice \"{value}\" (valid choices: " + string.Join(", ", dictionary.Keys) + ").");
            InternalValue = new Box<K>(value);
            return this;
        }
        private static K ParseDictionaryChoice(string str, Dictionary<K, String> dictionary)
        {
            try
            {
                K value = dictionary.First(kv => kv.Value == str).Key;
                return value;
            }
            catch (InvalidOperationException)
            {
                throw new OptionParseException($"Invalid choice \"{str}\" (valid choices: " + string.Join(", ", dictionary.Values) + ").");
            }
        }
    }

    public class HelpOption : Option<Null>
    {
        public HelpOption(Options options, params string[] names) : base(0, args => new Null(), names)
        {
            this.InternalValue = new Null();
            this.AddAction(options.PrintHelp);
        }
    }


    public class Argument
    {
        public String Name { get; private set; }
        public List<String> Helps { get; private set; }

        public Argument(String name)
        {
            this.Name = name;
            this.Helps = new List<string>();
        }

        public Argument AddHelp(String helpMessage)
        {
            Helps.Add(helpMessage);
            return this;
        }

    }

    class Queue<T>
    {
        private T[] items;
        private int offset;
        public int Count => items.Length - offset;

        public Queue(IEnumerable<T> items)
        {
            this.items = items.ToArray();
            this.offset = 0;
        }

        public T[] Pop(int count)
        {
            if (this.Count < count) throw new ArgumentException($"Cannot pop {count} elements from {this.Count}");
            T[] result = items[offset..(offset + count)];
            offset += count;
            return result;
        }

        public T Pop()
        {
            return Pop(1)[0];
        }
    }

}

