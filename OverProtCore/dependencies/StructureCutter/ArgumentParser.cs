using System;
using System.Collections.Generic;
using System.Linq;

namespace StructureCutter
{
    public class ArgParseException : Exception
    {
        public ArgParseException(string message) : base(message) { }
        public ArgParseException(string message, Exception innerException) : base(message, innerException) { }
    }

    public class Null { }


    public class ArgumentParser
    {
        public string GlobalHelp { get; set; }
        public List<IArgument> Arguments { get; private set; }
        public List<IArgument> Options { get; private set; }
        private Dictionary<string, IArgument> OptionDict { get; }

        public ArgumentParser(string globalHelp = null)
        {
            this.GlobalHelp = globalHelp;
            this.Arguments = new List<IArgument>();
            this.Options = new List<IArgument>();
            this.OptionDict = new Dictionary<string, IArgument>();
            this.AddHelpArg("-h", "--help").AddHelp("Print this help message and exit.");
        }

        public Argument<T> AddArg<T>(Argument<T> argument)
        {
            foreach (string name in argument.Names)
            {
                if (OptionDict.ContainsKey(name))
                    throw new ArgumentException($"Option conflict: multiple definitions for option {name}");
                OptionDict[name] = (IArgument)argument;
            }
            if (argument.Type == ArgumentType.Argument)
                this.Arguments.Add((IArgument)argument);
            else
                this.Options.Add((IArgument)argument);
            return argument;
        }
        public Argument<T> AddArg<T>(int numParams, Func<string[], T> parser, params string[] names) => this.AddArg(new Argument<T>(numParams, parser, names));
        public Argument<bool> AddSwitchArg(params string[] names) => this.AddArg(Argument<bool>.NewSwitchArg(names));
        public Argument<int> AddIntArg(params string[] names) => this.AddArg(Argument<int>.NewIntArg(names));
        public Argument<double> AddDoubleArg(params string[] names) => this.AddArg(Argument<double>.NewDoubleArg(names));
        public Argument<string> AddStringArg(params string[] names) => this.AddArg(Argument<string>.NewStringArg(names));
        public Argument<Null> AddHelpArg(params string[] names) => this.AddArg(Argument<Null>.NewHelpArg(this, names));
        public Argument<K> AddDictionaryChoiceArg<K>(Dictionary<K, string> dictionary, params string[] names) => this.AddArg(Argument<K>.NewDictionaryChoiceArg(dictionary, names));

        private void ParseOption(IArgument option, string optionName, Queue<string> argQueue)
        {
            if (argQueue.Count < option.NumParams)
            {
                string message = option.NumParams == 1 ? "Missing value." : $"Requires {option.NumParams} values, got {argQueue.Count}.";
                throw new ArgParseException(message);
            }
            string[] optionArgs = argQueue.Pop(option.NumParams);
            foreach (var cm in option.Constraints)
                if (!cm.constraint(optionArgs))
                    throw new ArgParseException("\"" + string.Join(" ", optionArgs) + "\", " + cm.message);
            option.SetValue(optionArgs);
            foreach (var action in option.Actions)
                action();
        }

        public bool TryParse(IEnumerable<string> args)
        {
            string[] extraArgs;
            bool ok = TryParse(args, out extraArgs);
            if (!ok) return false;
            if (extraArgs.Length > 0)
            {
                Lib.PrintError($"Too many arguments: " + string.Join(' ', extraArgs));
                Lib.PrintError($"Run   dotnet {System.AppDomain.CurrentDomain.FriendlyName}.dll --help   for usage information.");
                return false;
            }
            return true;
        }
        public bool TryParse(IEnumerable<string> args, out string[] extraArgs)
        {
            extraArgs = null;
            Queue<string> argQueue = new Queue<string>(args);
            List<string> remainingArgs = new List<string>();
            while (argQueue.Count > 0)
            {
                string name = argQueue.Pop();
                if (name.Length > 0 && name[0] == '-')
                {
                    if (OptionDict.ContainsKey(name))
                    {
                        IArgument option = OptionDict[name];
                        try
                        {
                            ParseOption(option, name, argQueue);
                        }
                        catch (ArgParseException e)
                        {
                            Lib.PrintError($"Option {name}: {e.Message}");
                            Lib.PrintError($"Run   dotnet {System.AppDomain.CurrentDomain.FriendlyName}.dll --help   for usage information.");
                            return false;
                        }
                    }
                    else
                    {
                        Lib.PrintError($"Unknown option: {name}.");
                        Lib.PrintError($"Run   dotnet {System.AppDomain.CurrentDomain.FriendlyName}.dll --help   for usage information.");
                        return false;
                    }
                }
                else
                {
                    remainingArgs.Add(name);
                }
            }
            argQueue = new Queue<string>(remainingArgs);
            foreach (var argument in Arguments)
            {
                try
                {
                    ParseOption(argument, argument.Names[0], argQueue);
                }
                catch (ArgParseException e)
                {
                    Lib.PrintError($"Argument {argument.Names[0]}: {e.Message}");
                    Lib.PrintError($"Run   dotnet {System.AppDomain.CurrentDomain.FriendlyName}.dll --help   for usage information.");
                    return false;
                }
            }
            extraArgs = argQueue.Pop(argQueue.Count);
            return true;
        }

        public void PrintHelp()
        {
            const string OPTION_INDENT = "   ";
            const string OPTION_HELP_INDENT = "          ";
            if (GlobalHelp != null)
            {
                Console.WriteLine(GlobalHelp);
                Console.WriteLine();
            }
            Console.WriteLine("Usage:");
            Console.WriteLine($"{OPTION_INDENT}dotnet {System.AppDomain.CurrentDomain.FriendlyName}.dll [options] " + string.Join(" ", Arguments.Select(a => a.Names[0])));
            if (Arguments.Count > 0)
            {
                Console.WriteLine("\nArguments:");
                foreach (IArgument argument in Arguments)
                {
                    string line = OPTION_INDENT + argument.Names[0];
                    Lib.WriteInColor(ConsoleColor.Cyan, line);
                    foreach (string help in argument.Helps)
                        Console.WriteLine(OPTION_HELP_INDENT + help);
                }
            }
            if (Options.Count > 0)
            {
                Console.WriteLine("\nOptions:");
                foreach (IArgument option in Options)
                {
                    string line = OPTION_INDENT + string.Join(", ", option.Names) + " " + string.Join(" ", option.Parameters);
                    Lib.WriteInColor(ConsoleColor.Cyan, line);
                    foreach (string help in option.Helps)
                        Console.WriteLine(OPTION_HELP_INDENT + help);
                }
            }
            Environment.Exit(0);
        }

    }


    public interface IArgument
    {
        string[] Names { get; }
        int NumParams { get; }
        List<Action> Actions { get; }
        List<(Func<string[], bool> constraint, string message)> Constraints { get; }
        string[] Parameters { get; }
        List<string> Helps { get; }
        public void SetValue(string[] args);
    }

    public enum ArgumentType { Argument, Option }

    public class Argument<T> : IArgument
    {
        public string[] Names { get; private set; }
        public ArgumentType Type { get; private set; }
        public int NumParams { get; private set; }
        private Func<string[], T> Parser { get; set; }
        private List<T> AllowedValues { get; set; }
        public List<Action> Actions { get; private set; }
        public List<(Func<string[], bool>, string)> Constraints { get; private set; }
        public List<(Func<T, bool> constraint, string message)> ConstraintsX { get; private set; }
        public string[] Parameters { get; private set; }
        public List<string> Helps { get; private set; }
        private T _value;
        public T Value
        {
            get { return _value; }
            set
            {
                foreach (var cm in ConstraintsX)
                {
                    if (!cm.constraint(value))
                        throw new ArgParseException($"Invalid value. {cm.message}");
                }
                _value = CheckChoice(value, AllowedValues);
            }
        }
        public void SetValue(string[] args)
        {
            Value = Parser(args);
        }

        public Argument(int numParams, Func<string[], T> parser, params string[] names)
        {
            if (names.Length == 0)
                throw new ArgumentException("There must be at least one argument/option name.");
            if (names.Any(name => name == null || name.Length == 0))
                throw new ArgumentException("Argument/option names must not be empty or null.");
            if (names[0][0] == '-')
            {
                Type = ArgumentType.Option;
                if (names.Any(name => name[0] != '-'))
                    throw new ArgumentException("Option names must begin with -.");
            }
            else
            {
                Type = ArgumentType.Argument;
                if (names.Length > 1)
                    throw new ArgumentException("Positional argument can only have one name.");
            }
            Names = names;
            NumParams = numParams;
            Parser = parser;
            Actions = new List<Action>();
            Constraints = new List<(Func<string[], bool>, string)>();
            ConstraintsX = new List<(Func<T, bool>, string)>();
            Parameters = Enumerable.Repeat(names[0].TrimStart('-').ToUpper(), numParams).ToArray();
            Helps = new List<string>();
            _value = default(T);
        }

        public Argument<T> SetChoices(params T[] choices)
        {
            AllowedValues = choices.ToList();
            return this;
        }
        public Argument<T> SetDefault(T _value)
        {
            try { Value = _value; }
            catch (ArgParseException ex) { throw new ArgumentException($"{Type} {Names[0]}: {ex.Message}"); }
            return this;
        }

        public Argument<T> AddAction(Action action)
        {
            this.Actions.Add(action);
            return this;
        }

        public Argument<T> AddConstraint(Func<string[], bool> constraint, string errorMessage)
        {
            Constraints.Add((constraint, errorMessage));
            return this;
        }
        public Argument<T> AddConstraintX(Func<T, bool> constraint, string errorMessage)
        {
            ConstraintsX.Add((constraint, errorMessage));
            return this;
        }

        public Argument<T> NameParameters(params string[] parameterNames)
        {
            Parameters = parameterNames;
            return this;
        }

        public Argument<T> AddHelp(string helpMessage)
        {
            Helps.Add(helpMessage);
            return this;
        }

        public override string ToString() => $"{Type} {Names[0]} [{Value}]";

        public static Argument<bool> NewSwitchArg(params string[] names) => new Argument<bool>(0, args => true, names);
        public static Argument<int> NewIntArg(params string[] names) => new Argument<int>(1, args => ParseInt(args[0]), names);
        public static Argument<double> NewDoubleArg(params string[] names) => new Argument<double>(1, args => ParseDouble(args[0]), names);
        public static Argument<string> NewStringArg(params string[] names) => new Argument<string>(1, args => args[0], names);
        public static Argument<Null> NewHelpArg(ArgumentParser argParser, params string[] names) => new Argument<Null>(0, args => new Null(), names).AddAction(argParser.PrintHelp);
        public static Argument<K> NewDictionaryChoiceArg<K>(Dictionary<K, string> dictionary, params string[] names)
        {
            if (dictionary.Count == 0)
                throw new ArgumentException($"Argument/option {names[0]}: dictionary must have at least one key-value pair.");
            return new Argument<K>(1, args => ParseDictionaryChoice(args[0], dictionary), names)
                .SetChoices(dictionary.Keys.ToArray())
                .SetDefault(dictionary.First().Key);
        }

        private static int ParseInt(string str)
        {
            try { return Int32.Parse(str); }
            catch { throw new ArgParseException($"Could not parse \"{str}\" as integer."); }
        }
        private static double ParseDouble(string str)
        {
            try { return Double.Parse(str); }
            catch { throw new ArgParseException($"Could not parse \"{str}\" as float."); }
        }
        private static K ParseDictionaryChoice<K>(string str, Dictionary<K, string> dictionary)
        {
            try { return dictionary.First(kv => kv.Value == str).Key; }
            catch (InvalidOperationException) { throw new ArgParseException($"Invalid choice \"{str}\" (valid choices: " + string.Join(", ", dictionary.Values) + ")."); }
        }
        private static K CheckChoice<K>(K value, List<K> choices)
        {
            if (choices == null || choices.Contains(value))
                return value;
            else
                throw new ArgParseException($"Invalid choice \"{value}\" (valid choices: " + string.Join(", ", choices) + ").");
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

        public T Pop() => Pop(1)[0];
    }

}

