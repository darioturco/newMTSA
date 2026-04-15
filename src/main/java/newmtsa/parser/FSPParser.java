package newmtsa.parser;

import newmtsa.parser.ast.*;
import org.antlr.v4.runtime.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class FSPParser {

    public static FSPModel parse(Path file) throws IOException {
        // Pre-process: expand macros before feeding source to ANTLR.
        MacroPreprocessor.Result preprocessed =
                MacroPreprocessor.process(Files.readString(file));

        CharStream chars = CharStreams.fromString(preprocessed.source());

        FSPGrammarLexer lexer = new FSPGrammarLexer(chars);
        CommonTokenStream tokens = new CommonTokenStream(lexer);
        FSPGrammarParser parser = new FSPGrammarParser(tokens);

        List<String> errors = new ArrayList<>();
        BaseErrorListener errorCollector = new BaseErrorListener() {
            @Override
            public void syntaxError(Recognizer<?, ?> recognizer, Object offendingSymbol,
                                    int line, int charPositionInLine,
                                    String msg, RecognitionException e) {
                errors.add("line " + line + ":" + charPositionInLine + " " + msg);
            }
        };
        lexer.removeErrorListeners();
        lexer.addErrorListener(errorCollector);
        parser.removeErrorListeners();
        parser.addErrorListener(errorCollector);

        FSPGrammarParser.FspDescriptionContext tree = parser.fspDescription();

        Visitor visitor = new Visitor();
        visitor.visit(tree);

        // After the visitor has resolved all constants, substitute their names
        // with numeric values in macro bodies and LTS action/state labels.
        Map<String, Integer> env = visitor.constEnv;

        List<MacroDef> resolvedMacros = preprocessed.macros().stream()
                .map(m -> new MacroDef(m.name(), m.params(),
                        substituteConsts(m.body(), env, m.params())))
                .collect(Collectors.toList());

        // Processes are now instantiated on demand during composite visits;
        // the instances map holds the results in insertion order.
        List<LTS> resolvedProcesses = visitor.instances.values().stream()
                .map(lts -> new LTS(
                        lts.name(), lts.initialState(),
                        lts.states().stream()
                                .map(s -> substituteConsts(s, env, List.of()))
                                .collect(Collectors.toList()),
                        lts.actions().stream()
                                .map(a -> substituteConsts(a, env, List.of()))
                                .distinct().collect(Collectors.toList()),
                        lts.transitions().stream()
                                .map(t -> new Transition(
                                        substituteConsts(t.from(), env, List.of()),
                                        substituteConsts(t.action(), env, List.of()),
                                        substituteConsts(t.to(), env, List.of())))
                                .collect(Collectors.toList()),
                        false, Set.of()))
                .collect(Collectors.toList());

        Set<String> actions = resolvedProcesses.stream()
                .flatMap(lts -> lts.actions().stream())
                .collect(Collectors.toSet());

        // Resolve assert aliases in liveness/safety/assumption:
        // if "assert A = F" and liveness = {A}, replace A with F.
        Map<String, String> assertAliases = visitor.simpleAssertMap;
        List<ControllerSpecDef> resolvedSpecs = visitor.controllerSpecs.stream()
                .map(cs -> new ControllerSpecDef(
                        cs.name(),
                        resolveAssertNames(cs.liveness(),   assertAliases),
                        resolveAssertNames(cs.safety(),     assertAliases),
                        resolveAssertNames(cs.assumption(), assertAliases),
                        cs.controllable(),
                        cs.marking(),
                        cs.nonblocking()))
                .collect(Collectors.toList());

        return new FSPModel(
                List.copyOf(resolvedProcesses),
                List.copyOf(visitor.composites),
                List.copyOf(visitor.fluents),
                List.copyOf(visitor.asserts),
                List.copyOf(visitor.ltlProperties),
                List.copyOf(visitor.sets),
                List.copyOf(resolvedSpecs),
                List.copyOf(visitor.constants),
                List.copyOf(visitor.ranges),
                List.copyOf(resolvedMacros),
                Set.copyOf(actions),
                List.copyOf(errors)
        );
    }

    // ── model display ─────────────────────────────────────────────────────────

    /**
     * Print a human-readable summary of all sections of a parsed {@link FSPModel}
     * to standard output.
     */
    public static void printModel(FSPModel model) {
        printSection("Constants", model.constants(),
                c -> c.name() + " = " + c.value());

        printSection("Ranges", model.ranges(),
                r -> r.name() + " = " + r.init() + ".." + r.end()
                        + "  (size: " + r.size() + ")");

        printSection("Sets", model.sets(),
                s -> s.name() + " = {" + String.join(", ", s.elements()) + "}");

        printSection("Macros", model.macros(),
                m -> m.name() + "(" + String.join(", ", m.params()) + ")"
                        + " = " + m.body());

        printSection("Actions", java.util.List.of(
                model.actions().stream().sorted()
                     .collect(java.util.stream.Collectors.joining(", ", "{ ", " }"))
        ), a -> a);

        printProcesses(model.processes());

        printSection("Composites (ParallelCompositionLazy)", model.composites(),
                c -> c.name() + "  components: ["
                        + c.components().stream().map(LTS::name)
                                .reduce((a, b) -> a + ", " + b).orElse("unresolved")
                        + "]");

        printSection("Fluents", model.fluents(), f -> {
            java.util.List<String> init = f.transitions().stream()
                    .filter(t -> t.from().equals("off") && t.to().equals("on"))
                    .map(Transition::action).collect(java.util.stream.Collectors.toList());
            java.util.List<String> term = f.transitions().stream()
                    .filter(t -> t.from().equals("on") && t.to().equals("off"))
                    .map(Transition::action).collect(java.util.stream.Collectors.toList());
            return f.name() + " = <{" + String.join(", ", init) + "}, {" + String.join(", ", term) + "}>";
        });

        printSection("Asserts", model.asserts(), p -> {
            String fluentList = p.fluents().isEmpty() ? "" : "  fluents: " + p.fluents();
            java.util.List<String> init = p.lts().transitions().stream()
                    .filter(t -> t.from().equals("off") && t.to().equals("on"))
                    .map(Transition::action).collect(java.util.stream.Collectors.toList());
            java.util.List<String> term = p.lts().transitions().stream()
                    .filter(t -> t.from().equals("on") && t.to().equals("off"))
                    .map(Transition::action).collect(java.util.stream.Collectors.toList());
            return p.name() + " = <{" + String.join(", ", init) + "}, {" + String.join(", ", term) + "}>"
                    + fluentList;
        });

        printSection("LTL Properties", model.ltlProperties(), LtlPropertyDef::name);

        printSection("Controller Specs", model.controllerSpecs(), cs -> {
            StringBuilder sb = new StringBuilder(cs.name()).append(" {\n");
            if (!cs.liveness().isEmpty())
                sb.append("    liveness     = ").append(cs.liveness()).append("\n");
            if (!cs.safety().isEmpty())
                sb.append("    safety       = ").append(cs.safety()).append("\n");
            if (!cs.assumption().isEmpty())
                sb.append("    assumption   = ").append(cs.assumption()).append("\n");
            if (!cs.controllable().isEmpty())
                sb.append("    controllable = ").append(cs.controllable()).append("\n");
            if (!cs.marking().isEmpty())
                sb.append("    marking      = ").append(cs.marking()).append("\n");
            if (cs.nonblocking())
                sb.append("    nonblocking\n");
            sb.append("  }");
            return sb.toString();
        });

        if (!model.errors().isEmpty()) {
            System.out.println("\n--- Parse Errors (" + model.errors().size() + ") ---");
            model.errors().forEach(e -> System.out.println("  " + e));
        }
    }

    private static void printProcesses(java.util.List<LTS> processes) {
        System.out.println("\n--- Processes / LTS (" + processes.size() + ") ---");
        for (LTS p : processes) {
            System.out.println("\n  " + p.name() + "  (initial: " + p.initialState() + ")");
            System.out.println("    States (" + p.states().size() + "): "
                    + String.join(", ", p.states()));
            System.out.println("    Actions (" + p.actions().size() + "): "
                    + String.join(", ", p.actions()));
            System.out.println("    Transitions (" + p.transitions().size() + "):");
            for (Transition t : p.transitions()) {
                System.out.println("      " + t.from() + " --[" + t.action() + "]--> " + t.to());
            }
        }
    }

    private static <T> void printSection(String title, java.util.List<T> items,
                                         java.util.function.Function<T, String> fmt) {
        System.out.println("\n--- " + title + " (" + items.size() + ") ---");
        if (items.isEmpty()) {
            System.out.println("  (none)");
        } else {
            items.forEach(item -> System.out.println("  " + fmt.apply(item)));
        }
    }

    // ── assert resolution ─────────────────────────────────────────────────────

    /** Replaces each name in {@code names} with its assert alias target if one exists. */
    private static List<String> resolveAssertNames(List<String> names, Map<String, String> aliases) {
        return names.stream()
                .map(n -> aliases.getOrDefault(n, n))
                .collect(Collectors.toList());
    }

    /**
     * Replace whole-word occurrences of every constant name in {@code text}
     * with its resolved integer value, then fold any all-integer arithmetic
     * sub-expressions that result (e.g. {@code 2-1} → {@code 1}).
     * Names listed in {@code exclude} are left untouched (macro parameters).
     * Constants are sorted longest-first to prevent a short name (e.g. "N")
     * from being replaced inside a longer one (e.g. "Nodes").
     */
    private static String substituteConsts(String text,
                                           Map<String, Integer> env,
                                           List<String> exclude) {
        // Process longer names first to avoid partial replacements.
        List<Map.Entry<String, Integer>> sorted = env.entrySet().stream()
                .filter(e -> !exclude.contains(e.getKey()))
                .sorted(Comparator.comparingInt((Map.Entry<String, Integer> e) ->
                        e.getKey().length()).reversed())
                .collect(Collectors.toList());

        for (Map.Entry<String, Integer> e : sorted) {
            text = text.replaceAll(
                    "\\b" + Pattern.quote(e.getKey()) + "\\b",
                    Matcher.quoteReplacement(String.valueOf(e.getValue())));
        }
        return foldConsts(text);
    }

    // ── Arithmetic constant-folding ───────────────────────────────────────────
    // After const-name substitution the text may contain sub-expressions that
    // are now fully numeric, e.g. "2-1" or "(3+4)*2".  These are evaluated
    // using three passes (multiplication/division before addition/subtraction)
    // applied repeatedly until the text stabilises.
    //
    // Expressions that still contain variable names are left untouched.
    // The lookbehind (?<!\w) ensures we never start a number match in the
    // middle of an identifier (e.g. the "1" in "Fid1" is not matched).

    /** Matches:  integer  [* / % \]  integer  */
    private static final Pattern FOLD_MUL = Pattern.compile(
            "(?<!\\w)(-?\\d+)\\s*([*/%\\\\])\\s*(-?\\d+)");

    /** Matches:  integer  [+ -]  integer   but not inside a larger number */
    private static final Pattern FOLD_ADD = Pattern.compile(
            "(?<!\\w)(-?\\d+)\\s*([+\\-])\\s*(-?\\d+)(?!\\d)");

    /** Matches:  integer  comparison-op  integer */
    private static final Pattern FOLD_CMP = Pattern.compile(
            "(?<!\\w)(-?\\d+)\\s*(==|!=|<=|>=|<|>)\\s*(-?\\d+)(?!\\d)");

    /** Matches a fully-numeric ternary:  integer ? integer : integer */
    private static final Pattern FOLD_TERNARY = Pattern.compile(
            "(?<!\\w)(-?\\d+)\\s*\\?\\s*(-?\\d+)\\s*:\\s*(-?\\d+)(?!\\d)");

    /** Matches:  integer  &&  integer */
    private static final Pattern FOLD_AND = Pattern.compile(
            "(?<!\\w)(-?\\d+)\\s*&&\\s*(-?\\d+)(?!\\d)");

    /** Matches:  integer  ||  integer */
    private static final Pattern FOLD_OR = Pattern.compile(
            "(?<!\\w)(-?\\d+)\\s*\\|\\|\\s*(-?\\d+)(?!\\d)");

    private static String foldConsts(String text) {
        String prev;
        int limit = 40;   // guard against infinite loops
        do {
            prev = text;
            // Fold all arithmetic to a local fixed point FIRST so that the
            // branches of a ternary (e.g. 0-1 → -1) are fully numeric before
            // the ternary pattern is applied.  Without this, FOLD_TERNARY would
            // greedily match the leading digit of an un-folded expression like
            // "0-1" and produce a wrong result.
            String arith;
            // Fold all multiplications to convergence BEFORE any additions,
            // so that precedence is respected (e.g. 2*3+1*4 → 6+4 = 10, not 7*4 = 28).
            do { arith = text; text = foldOne(text, FOLD_MUL); } while (!text.equals(arith));
            do { arith = text; text = foldOne(text, FOLD_ADD); } while (!text.equals(arith));
            do { arith = text; text = foldCmp(text); } while (!text.equals(arith));
            do { arith = text; text = foldAnd(text); } while (!text.equals(arith));
            do { arith = text; text = foldOr(text);  } while (!text.equals(arith));
            text = foldTernary(text);             // n ? a : b → a or b
            // Remove redundant parens around a bare integer, but only when NOT
            // preceded by a word character (to avoid mangling names like "Monitor(0)").
            text = text.replaceAll("(?<!\\w)\\((-?\\d+)\\)", "$1");
        } while (!text.equals(prev) && --limit > 0);
        return text;
    }

    private static String foldAnd(String text) {
        Matcher m = FOLD_AND.matcher(text);
        if (!m.find()) return text;
        int r = (Integer.parseInt(m.group(1)) != 0 && Integer.parseInt(m.group(2)) != 0) ? 1 : 0;
        return text.substring(0, m.start()) + r + text.substring(m.end());
    }

    private static String foldOr(String text) {
        Matcher m = FOLD_OR.matcher(text);
        if (!m.find()) return text;
        int r = (Integer.parseInt(m.group(1)) != 0 || Integer.parseInt(m.group(2)) != 0) ? 1 : 0;
        return text.substring(0, m.start()) + r + text.substring(m.end());
    }

    private static String foldCmp(String text) {
        Matcher m = FOLD_CMP.matcher(text);
        if (!m.find()) return text;
        long left  = Long.parseLong(m.group(1));
        String op  = m.group(2);
        long right = Long.parseLong(m.group(3));
        int result = switch (op) {
            case "==" -> left == right ? 1 : 0;
            case "!=" -> left != right ? 1 : 0;
            case "<"  -> left <  right ? 1 : 0;
            case ">"  -> left >  right ? 1 : 0;
            case "<=" -> left <= right ? 1 : 0;
            case ">=" -> left >= right ? 1 : 0;
            default   -> throw new IllegalArgumentException(op);
        };
        return text.substring(0, m.start()) + result + text.substring(m.end());
    }

    private static String foldTernary(String text) {
        Matcher m = FOLD_TERNARY.matcher(text);
        if (!m.find()) return text;
        int cond   = Integer.parseInt(m.group(1));
        String val = cond != 0 ? m.group(2) : m.group(3);
        return text.substring(0, m.start()) + val + text.substring(m.end());
    }

    /** Apply {@code pattern} once (leftmost match) and return the updated text. */
    private static String foldOne(String text, Pattern pattern) {
        Matcher m = pattern.matcher(text);
        if (!m.find()) return text;
        long left  = Long.parseLong(m.group(1));
        String op  = m.group(2);
        long right = Long.parseLong(m.group(3));
        try {
            long result = switch (op) {
                case "+"  -> left + right;
                case "-"  -> left - right;
                case "*"  -> left * right;
                case "/"  -> left / right;
                case "%"  -> left % right;
                case "\\" -> left / right;   // FSP integer division
                default   -> throw new IllegalArgumentException(op);
            };
            return text.substring(0, m.start()) + result + text.substring(m.end());
        } catch (ArithmeticException e) {
            return text;   // division by zero — leave unchanged
        }
    }

    /**
     * A concrete action string plus any range-variable bindings introduced
     * by a labeled index in the action label (e.g. {@code [b:Area]} binds b).
     * The bindings must be active when the transition target is evaluated.
     */
    private record ActionExpansion(String text, Map<String, Integer> bindings) {}

    // ─── Visitor ──────────────────────────────────────────────────────────────
    // Walks the parse tree and collects top-level definitions into plain lists.
    // Each visitX method records the definition it cares about and returns null
    // (no deeper recursion needed — we only need top-level names and fields).

    private static final class Visitor extends FSPGrammarBaseVisitor<Void> {

        // Process templates: parse context stored on first visit, LTS built on demand.
        final Map<String, FSPGrammarParser.ProcessDefContext> templateCtx = new LinkedHashMap<>();
        // Instantiated LTS objects, keyed by instance name (e.g. "Philosopher(0)").
        // Insertion order matches the order composites first reference each instance.
        final Map<String, LTS>                 instances  = new LinkedHashMap<>();
        final List<ParallelCompositionLazy>    composites = new ArrayList<>();
        final List<LTS>               fluents         = new ArrayList<>();
        final List<LtlPropertyDef>    asserts         = new ArrayList<>();
        final List<LtlPropertyDef>    ltlProperties   = new ArrayList<>();
        final List<SetDef>            sets            = new ArrayList<>();
        final List<ControllerSpecDef> controllerSpecs = new ArrayList<>();
        final List<ConstDef>          constants       = new ArrayList<>();
        final List<RangeDef>          ranges          = new ArrayList<>();
        /** Maps assert name → single UPPER_ID it refers to (if assert is just an alias). */
        final Map<String, String>     simpleAssertMap = new LinkedHashMap<>();
        /** Maps fluent/assert name → its LTS, for use when building compound assert LTSes. */
        final Map<String, LTS>        propertyLTSMap  = new LinkedHashMap<>();

        // Resolved const values available for expression evaluation.
        // Keyed by base name so that later consts can reference earlier ones.
        private final Map<String, Integer> constEnv = new HashMap<>();

        // Tracks how many times each const base name has been seen,
        // used to produce unique names on repetition (N, N_1, N_2, …).
        private final Map<String, Integer> constNameCount = new HashMap<>();

        // Per-process environment: global consts merged with the current
        // process's parameter defaults (and any active range-loop variables).
        // Set at the start of visitProcessDef, reset when it returns.
        private Map<String, Integer> currentEnv = new HashMap<>();

        // Declared ranges keyed by name:  Area → [lo, hi]
        // Used when a local-process index spec references a range by name, e.g. [a:Area].
        private final Map<String, int[]> rangeMap = new HashMap<>();

        // Local-process aliases within the current LTS being built.
        // e.g.  Etiquete[0] = Ready  →  localAliases["Etiquete[0]"] = "Ready"
        // Populated in expandIndexSpec; followed in lpStateName.
        private final Map<String, String> localAliases = new HashMap<>();

        // Set during visitProcessDef so that lpStateName can resolve the process
        // name back to the actual initial state when it appears as a transition target.
        // e.g.  TU = Idle  →  currentProcessName="TU", currentProcessInitialState="Idle"
        //       Monitor = (eat -> Done)  →  both are null (process name IS the initial state)
        private String currentProcessName          = null;
        private String currentProcessInitialState  = null;

        // ── const ────────────────────────────────────────────────────────────
        // const N = 2    const Philosophers = N

        @Override
        public Void visitConstDef(FSPGrammarParser.ConstDefContext ctx) {
            String baseName = ctx.UPPER_ID().getText();
            int    value    = evalExpr(ctx.expr());

            // Update the environment so later consts can reference this one.
            constEnv.put(baseName, value);

            int count = constNameCount.merge(baseName, 1, Integer::sum);
            String uniqueName = count == 1 ? baseName : baseName + "_" + (count - 1);

            constants.add(new ConstDef(uniqueName, value));
            return null;
        }

        // ── range ────────────────────────────────────────────────────────────
        // range Phil = 0..Philosophers-1

        @Override
        public Void visitRangeDef(FSPGrammarParser.RangeDefContext ctx) {
            String name = ctx.UPPER_ID().getText();
            FSPGrammarParser.RangeOrExprContext roe = ctx.rangeOrExpr();

            if (roe.expr().size() < 2)
                throw new IllegalArgumentException(
                        "Range '" + name + "' requires two bounds separated by '..'");

            int init = evalExpr(roe.expr(0));
            int end  = evalExpr(roe.expr(1));   // throws if init > end via RangeDef compact ctor
            ranges.add(new RangeDef(name, init, end));
            rangeMap.put(name, new int[]{init, end});
            return null;
        }

        /**
         * Recursively evaluate an arithmetic expression using the resolved
         * constants collected so far.  Supports:
         *   INT literal, constant reference (UPPER_ID / LOWER_ID),
         *   unary minus, grouping, binary + - * / %, ternary ? :
         */
        private int evalExpr(FSPGrammarParser.ExprContext ctx) {
            // INT literal
            if (ctx.INT() != null) {
                return Integer.parseInt(ctx.INT().getText());
            }

            // Variable / constant reference  (anyId with no argument list)
            if (ctx.anyId() != null) {
                String name = ctx.anyId().getText();
                Integer resolved = constEnv.get(name);
                if (resolved == null)
                    throw new IllegalArgumentException(
                            "Undefined constant referenced in const expression: " + name);
                return resolved;
            }

            int subExprs = ctx.expr().size();

            // Unary:  '-' expr   or   '(' expr ')'
            if (subExprs == 1) {
                int inner = evalExpr(ctx.expr(0));
                return ctx.getChild(0).getText().equals("-") ? -inner : inner;
            }

            // Ternary:  expr '?' expr ':' expr
            if (subExprs == 3) {
                return evalExpr(ctx.expr(0)) != 0
                        ? evalExpr(ctx.expr(1))
                        : evalExpr(ctx.expr(2));
            }

            // Binary:  expr op expr
            if (subExprs == 2) {
                int left  = evalExpr(ctx.expr(0));
                int right = evalExpr(ctx.expr(1));
                String op = ctx.getChild(1).getText();
                return switch (op) {
                    case "+"  -> left + right;
                    case "-"  -> left - right;
                    case "*"  -> left * right;
                    case "/"  -> left / right;
                    case "%"  -> left % right;
                    case "\\" -> left / right;   // FSP integer division
                    default   -> throw new IllegalArgumentException(
                            "Unsupported operator in const expression: " + op);
                };
            }

            throw new IllegalArgumentException(
                    "Cannot evaluate const expression: " + ctx.getText());
        }

        // ── process (LTS) ────────────────────────────────────────────────────
        // Builds a full LTS: states, unique base actions, and transitions.
        // Action chains (a -> b -> S) are expanded with generated intermediate
        // states named _StateName_N to preserve one action per edge.

        @Override
        public Void visitProcessDef(FSPGrammarParser.ProcessDefContext ctx) {
            // Store only the parse context; the LTS will be built on demand when
            // the process is referenced from a composite (possibly with explicit args).
            templateCtx.put(ctx.UPPER_ID().getText(), ctx);
            return null;
        }

        // ── on-demand instantiation ───────────────────────────────────────────

        /**
         * Return (possibly creating) the LTS instance for {@code processName}
         * with the given positional argument values.
         * Instance name:  "Philosopher(0)"  for args=[0],  "TU" for args=[].
         */
        private LTS getOrInstantiate(String processName, List<Integer> args) {
            String instanceName = makeInstanceName(processName, args);
            if (instances.containsKey(instanceName)) return instances.get(instanceName);

            FSPGrammarParser.ProcessDefContext ctx = templateCtx.get(processName);
            if (ctx == null) return null;   // unknown process — composite reference?

            Map<String, Integer> paramEnv = buildParamEnv(ctx, args);
            LTS lts = buildLTSFromTemplate(ctx, instanceName, paramEnv);
            instances.put(instanceName, lts);
            return lts;
        }

        private String makeInstanceName(String processName, List<Integer> args) {
            if (args.isEmpty()) return processName;
            return processName + "(" +
                    args.stream().map(Object::toString).collect(Collectors.joining(", ")) + ")";
        }

        /**
         * Map template parameter names to the supplied argument values.
         * Missing arguments fall back to the parameter's default expression.
         */
        private Map<String, Integer> buildParamEnv(
                FSPGrammarParser.ProcessDefContext ctx, List<Integer> args) {
            Map<String, Integer> paramEnv = new LinkedHashMap<>();
            if (ctx.paramDefList() == null) return paramEnv;
            List<FSPGrammarParser.ParamDefContext> params = ctx.paramDefList().paramDef();
            for (int i = 0; i < params.size(); i++) {
                String pname = params.get(i).anyId().getText();
                if (i < args.size()) {
                    paramEnv.put(pname, args.get(i));
                } else if (params.get(i).expr() != null) {
                    try { paramEnv.put(pname, evalExpr(params.get(i).expr())); }
                    catch (IllegalArgumentException ignored) { }
                }
            }
            return paramEnv;
        }

        /**
         * Build a concrete LTS from a process template with the given parameter
         * bindings.  {@code instanceName} is the name of the resulting LTS
         * (e.g. "Philosopher(0)"); it may differ from the template name when
         * explicit args are supplied.
         */
        private LTS buildLTSFromTemplate(
                FSPGrammarParser.ProcessDefContext ctx,
                String instanceName,
                Map<String, Integer> paramEnv) {

            String templateName = ctx.UPPER_ID().getText();

            // Set per-instance environment (consts + params) and alias resolution fields.
            currentEnv = new HashMap<>(constEnv);
            currentEnv.putAll(paramEnv);
            currentProcessName         = templateName;
            currentProcessInitialState = null;

            Set<String>      states      = new LinkedHashSet<>();
            Set<String>      actions     = new LinkedHashSet<>();
            List<Transition> transitions = new ArrayList<>();
            int[]            counter     = {0};

            FSPGrammarParser.LocalProcessContext initLp = ctx.localProcess();
            if (initLp == null) {
                resetProcessState();
                return new LTS(instanceName, instanceName, List.of(), List.of(), List.of(), false, Set.of());
            }

            String initialState;
            if (initLp.UPPER_ID() != null) {
                // e.g.  TU = Idle   /   Philosopher(Pid=0) = Idle
                // Template name is an alias → map "templateName" back to this state.
                initialState = lpStateName(initLp);
                currentProcessInitialState = initialState;
                states.add(initialState);
            } else {
                // e.g.  Monitor(Id=0) = (eat[Id] -> Done)
                // The initial state is the instance itself.
                // Map template name → instance name so back-refs ("-> Monitor") resolve.
                initialState = instanceName;
                currentProcessInitialState = instanceName;
                states.add(instanceName);
                extractFromLocalProcess(instanceName, initLp, states, actions, transitions, counter);
            }

            for (var lpd : ctx.localProcessDef())
                expandLocalProcessDef(lpd, states, actions, transitions, counter);

            resetProcessState();
            return new LTS(instanceName, initialState,
                    List.copyOf(states), List.copyOf(actions), List.copyOf(transitions), false, Set.of());
        }

        private void resetProcessState() {
            currentEnv = new HashMap<>(constEnv);
            currentProcessName = null;
            currentProcessInitialState = null;
            localAliases.clear();
        }

        /**
         * State-reference name from a localProcess node (UPPER_ID + optional index exprs).
         * Applies currentEnv substitution so parameter names and constants are resolved.
         * e.g.  Etiquete[Steps]  with Steps=2  →  "Etiquete[2]"
         *
         * Also resolves back-references to the process name itself.
         * e.g.  in  TU = Idle, Testing = (... -> TU),  "TU" → "Idle"
         */
        private String lpStateName(FSPGrammarParser.LocalProcessContext ctx) {
            StringBuilder sb = new StringBuilder(ctx.UPPER_ID().getText());
            for (var e : ctx.expr()) sb.append("[").append(e.getText()).append("]");
            String result = substituteConsts(sb.toString(), currentEnv, List.of());
            // Resolve top-level process name alias (e.g. TU = Idle → "TU" → "Idle").
            if (currentProcessInitialState != null && result.equals(currentProcessName)) {
                return currentProcessInitialState;
            }
            // Resolve local process aliases (e.g. Etiquete[0] = Ready → "Etiquete[0]" → "Ready").
            // Follow the chain in case of multi-hop aliases.
            String aliased;
            while ((aliased = localAliases.get(result)) != null) {
                result = aliased;
            }
            return result;
        }

        /**
         * Expand a localProcessDef, iterating over any range-variable index specs.
         * e.g.  Etiquete[s:1..Steps]  with Steps=2  →  creates Etiquete[1] and Etiquete[2].
         * Concrete index specs  [0]  produce a single state directly.
         */
        private void expandLocalProcessDef(
                FSPGrammarParser.LocalProcessDefContext lpd,
                Set<String> states, Set<String> actions,
                List<Transition> transitions, int[] counter) {
            expandIndexSpec(lpd.UPPER_ID().getText(), lpd.indexSpec(), 0,
                            new StringBuilder(), lpd.localProcess(),
                            states, actions, transitions, counter);
        }

        /**
         * Recursively resolve each index spec, then register the resulting state.
         * Range-variable specs  [s:lo..hi]  iterate over every value in [lo, hi];
         * concrete specs  [expr]  are evaluated once using currentEnv.
         */
        private void expandIndexSpec(
                String baseName,
                List<FSPGrammarParser.IndexSpecContext> specs,
                int specIdx,
                StringBuilder nameAccum,
                FSPGrammarParser.LocalProcessContext body,
                Set<String> states, Set<String> actions,
                List<Transition> transitions, int[] counter) {

            if (specIdx == specs.size()) {
                String stateName = baseName + nameAccum;
                if (body.UPPER_ID() != null) {
                    // Pure alias:  Etiquete[0] = Ready
                    // Register the alias; do NOT add a spurious state.
                    String target = lpStateName(body);
                    localAliases.put(stateName, target);
                } else {
                    states.add(stateName);
                    extractFromLocalProcess(stateName, body, states, actions, transitions, counter);
                }
                return;
            }

            FSPGrammarParser.IndexSpecContext spec = specs.get(specIdx);

            if (spec.anyId() != null) {
                // Range-variable binding:  [s:1..Steps]  or  [a:Area]
                String varName = spec.anyId().getText();
                var roe = spec.rangeOrExpr();

                int lo, hi;
                if (roe.expr().size() >= 2) {
                    // Explicit range:  s:1..Steps  — evaluate both bounds
                    lo = parseSubstituted(roe.expr(0).getText());
                    hi = parseSubstituted(roe.expr(1).getText());
                } else {
                    // Single expr:  a:Area  — could be a range name or a scalar
                    String boundText = substituteConsts(roe.expr(0).getText(), currentEnv, List.of());
                    int[] parsed = parseRangeBound(boundText);
                    if (parsed == null) {
                        // Unresolvable — keep raw spec text, no expansion
                        String rawSpec = spec.getText();
                        expandIndexSpec(baseName, specs, specIdx + 1,
                                        new StringBuilder(nameAccum).append("[").append(rawSpec).append("]"),
                                        body, states, actions, transitions, counter);
                        return;
                    }
                    lo = parsed[0];
                    hi = parsed[1];
                }

                Integer saved = currentEnv.get(varName);
                for (int v = lo; v <= hi; v++) {
                    currentEnv.put(varName, v);
                    expandIndexSpec(baseName, specs, specIdx + 1,
                                    new StringBuilder(nameAccum).append("[").append(v).append("]"),
                                    body, states, actions, transitions, counter);
                }
                // Restore the variable binding that existed before (or remove it).
                if (saved != null) currentEnv.put(varName, saved);
                else              currentEnv.remove(varName);

            } else {
                // Concrete expression:  [0]  or  [Pid]  — evaluate and continue.
                String val = substituteConsts(spec.expr().getText(), currentEnv, List.of());
                expandIndexSpec(baseName, specs, specIdx + 1,
                                new StringBuilder(nameAccum).append("[").append(val).append("]"),
                                body, states, actions, transitions, counter);
            }
        }

        /**
         * Substitute currentEnv into {@code text} (const-folding included)
         * and parse the result as an integer.  Used for range bounds.
         */
        private int parseSubstituted(String text) {
            return Integer.parseInt(substituteConsts(text, currentEnv, List.of()));
        }

        /**
         * Parse a range-bound string to a [lo, hi] pair.
         * <ul>
         *   <li>A plain integer → [n, n]</li>
         *   <li>A declared range name (e.g. "Area") → [range.lo, range.hi]</li>
         *   <li>Anything else → null (caller should fall back)</li>
         * </ul>
         */
        private int[] parseRangeBound(String text) {
            try {
                int v = Integer.parseInt(text);
                return new int[]{v, v};
            } catch (NumberFormatException e) {
                int[] bounds = rangeMap.get(text);
                return bounds;   // null if not found
            }
        }

        /** Extract transitions from a localProcess into the provided collections. */
        private void extractFromLocalProcess(
                String fromState,
                FSPGrammarParser.LocalProcessContext lp,
                Set<String> states, Set<String> actions,
                List<Transition> transitions, int[] counter) {
            if (lp.UPPER_ID() != null) return; // pure alias — no outgoing transitions
            if (lp.IF() != null) {
                // if (cond) then lp0 else lp1 — evaluate condition and recurse into branch.
                String cond = substituteConsts(lp.expr(0).getText(), currentEnv, List.of());
                FSPGrammarParser.LocalProcessContext branch =
                        "0".equals(cond) ? lp.localProcess(1) : lp.localProcess(0);
                extractFromLocalProcess(fromState, branch, states, actions, transitions, counter);
                return;
            }
            if (lp.choice() == null) return;
            extractFromChoice(fromState, lp.choice(), states, actions, transitions, counter);
        }

        private void extractFromChoice(
                String fromState,
                FSPGrammarParser.ChoiceContext choice,
                Set<String> states, Set<String> actions,
                List<Transition> transitions, int[] counter) {
            if (choice.FOREACH() != null) {
                // foreach [varName:range] innerChoice — expand for each value in range
                String varName = choice.LOWER_ID().getText();
                var roe = choice.rangeOrExpr();
                int lo, hi;
                if (roe.expr().size() >= 2) {
                    lo = Integer.parseInt(substituteConsts(roe.expr(0).getText(), currentEnv, List.of()));
                    hi = Integer.parseInt(substituteConsts(roe.expr(1).getText(), currentEnv, List.of()));
                } else {
                    String bound = substituteConsts(roe.expr(0).getText(), currentEnv, List.of());
                    int[] bounds = parseRangeBound(bound);
                    if (bounds == null) return;
                    lo = bounds[0]; hi = bounds[1];
                }
                Integer saved = currentEnv.get(varName);
                for (int v = lo; v <= hi; v++) {
                    currentEnv.put(varName, v);
                    extractFromChoice(fromState, choice.choice(), states, actions, transitions, counter);
                }
                if (saved == null) currentEnv.remove(varName); else currentEnv.put(varName, saved);
                return;
            }
            for (var gp : choice.guardedPrefix())
                extractFromGuardedPrefix(fromState, gp, states, actions, transitions, counter);
        }

        /**
         * Turn one guardedPrefix into one or more transitions.
         * Handles:
         *  - when-guards (skip if condition folds to "0")
         *  - range-indexed action labels (cat[0..Cats-1].move[b:Area] → multiple transitions)
         *  - if-then-else in the target localProcess
         *  - action chains  a -> b -> State
         */
        private void extractFromGuardedPrefix(
                String fromState,
                FSPGrammarParser.GuardedPrefixContext gp,
                Set<String> states, Set<String> actions,
                List<Transition> transitions, int[] counter) {

            // Evaluate the when-guard; skip if the condition is false (folds to "0").
            if (gp.WHEN() != null) {
                String cond = substituteConsts(gp.expr().getText(), currentEnv, List.of());
                if ("0".equals(cond)) return;
            }

            var ap = gp.actionPrefix();
            if (ap == null || ap.prefixActions() == null || ap.localProcess() == null)
                return; // parse error already recorded; skip this prefix

            extractChain(fromState, ap.prefixActions().processActionLabel(), 0,
                         ap.localProcess(), states, actions, transitions, counter);
        }

        /**
         * Recursively process a chain of action labels with range expansion.
         * For each expansion of label[labelIdx], temporarily binds its range
         * variables into currentEnv, then either recurses for the next label
         * or resolves the final target state.
         */
        private void extractChain(
                String fromState,
                List<FSPGrammarParser.ProcessActionLabelContext> labels,
                int labelIdx,
                FSPGrammarParser.LocalProcessContext finalTarget,
                Set<String> states, Set<String> actions,
                List<Transition> transitions, int[] counter) {

            var pal = labels.get(labelIdx);
            boolean isLast = (labelIdx == labels.size() - 1);

            for (ActionExpansion exp : expandProcessActionLabel(pal)) {
                actions.add(exp.text());

                // Temporarily bind any range variables from this label.
                Map<String, Integer> saved = new HashMap<>();
                for (var entry : exp.bindings().entrySet()) {
                    saved.put(entry.getKey(), currentEnv.get(entry.getKey()));
                    currentEnv.put(entry.getKey(), entry.getValue());
                }

                if (isLast) {
                    String next = resolveTargetState(fromState, finalTarget,
                                                     states, actions, transitions, counter);
                    if (next != null)
                        transitions.add(new Transition(fromState, exp.text(), next));
                } else {
                    String intermediate = "_" + fromState + "_" + counter[0]++;
                    states.add(intermediate);
                    transitions.add(new Transition(fromState, exp.text(), intermediate));
                    extractChain(intermediate, labels, labelIdx + 1, finalTarget,
                                 states, actions, transitions, counter);
                }

                // Restore bindings.
                for (String varName : exp.bindings().keySet()) {
                    Integer prev = saved.get(varName);
                    if (prev == null) currentEnv.remove(varName);
                    else currentEnv.put(varName, prev);
                }
            }
        }

        /**
         * Resolve a localProcess to a concrete state name, handling:
         *  - UPPER_ID [expr]*  →  state reference (via lpStateName)
         *  - if-then-else      →  evaluate condition, pick branch, recurse
         *  - '(' choice ')'    →  create intermediate state, extract choice
         */
        private String resolveTargetState(
                String fromState,
                FSPGrammarParser.LocalProcessContext lp,
                Set<String> states, Set<String> actions,
                List<Transition> transitions, int[] counter) {

            if (lp.UPPER_ID() != null) {
                String name = lpStateName(lp);
                states.add(name);  // ensures ERROR / STOP / etc. appear as states
                return name;
            }
            if (lp.IF() != null) {
                String cond = substituteConsts(lp.expr(0).getText(), currentEnv, List.of());
                // "0" = false → ELSE branch; anything else → THEN branch
                FSPGrammarParser.LocalProcessContext branch =
                        "0".equals(cond) ? lp.localProcess(1) : lp.localProcess(0);
                return resolveTargetState(fromState, branch, states, actions, transitions, counter);
            }
            if (lp.choice() != null) {
                String inter = "_" + fromState + "_" + counter[0]++;
                states.add(inter);
                extractFromChoice(inter, lp.choice(), states, actions, transitions, counter);
                return inter;
            }
            return null;
        }

        // ── Action-label expansion ────────────────────────────────────────────
        // Expand a processActionLabel (possibly containing range indices or
        // range-variable bindings) into a list of ActionExpansion records,
        // each holding a concrete action string and any variables bound during
        // the expansion (e.g. b=3 from [b:Area]).

        private List<ActionExpansion> expandProcessActionLabel(
                FSPGrammarParser.ProcessActionLabelContext pal) {
            if (pal.labelBase() != null) return expandActionLabelBase(pal.labelBase());
            // Set form: {a, b, c} — expand each child and concatenate.
            List<ActionExpansion> result = new ArrayList<>();
            for (var child : pal.processActionLabel())
                result.addAll(expandProcessActionLabel(child));
            return result;
        }

        private List<ActionExpansion> expandActionLabelBase(
                FSPGrammarParser.LabelBaseContext lb) {
            // Cartesian product over segments, joined with ".".
            List<ActionExpansion> acc = List.of(new ActionExpansion("", Map.of()));
            boolean first = true;
            for (var seg : lb.labelSegment()) {
                String sep = first ? "" : ".";
                first = false;
                List<ActionExpansion> next = new ArrayList<>();
                for (ActionExpansion prefix : acc) {
                    for (ActionExpansion s : expandActionLabelSegment(seg)) {
                        Map<String, Integer> merged = new HashMap<>(prefix.bindings());
                        merged.putAll(s.bindings());
                        next.add(new ActionExpansion(
                                prefix.text().isEmpty() ? s.text() : prefix.text() + sep + s.text(),
                                merged));
                    }
                }
                acc = next;
            }
            return acc;
        }

        private List<ActionExpansion> expandActionLabelSegment(
                FSPGrammarParser.LabelSegmentContext seg) {
            String base = seg.LOWER_ID().getText();
            if (seg.labelIndex().isEmpty()) return List.of(new ActionExpansion(base, Map.of()));
            // Cartesian product of index expansions, prefixed with base.
            List<ActionExpansion> indexPart = List.of(new ActionExpansion("", Map.of()));
            for (var idx : seg.labelIndex()) {
                List<ActionExpansion> next = new ArrayList<>();
                for (ActionExpansion prefix : indexPart) {
                    for (ActionExpansion s : expandActionLabelIndex(idx)) {
                        Map<String, Integer> merged = new HashMap<>(prefix.bindings());
                        merged.putAll(s.bindings());
                        next.add(new ActionExpansion(prefix.text() + s.text(), merged));
                    }
                }
                indexPart = next;
            }
            return indexPart.stream()
                    .map(e -> new ActionExpansion(base + e.text(), e.bindings()))
                    .collect(Collectors.toList());
        }

        /**
         * Expand one label index bracket to a list of "[value]" ActionExpansions.
         * Variable bindings (b:Area) are captured; plain ranges produce no binding.
         */
        private List<ActionExpansion> expandActionLabelIndex(
                FSPGrammarParser.LabelIndexContext idx) {
            if (idx.anyId() != null) {
                // Variable binding:  b:Area  — expand range, bind variable name.
                String varName = idx.anyId().getText();
                var roe = idx.rangeOrExpr();
                int lo, hi;
                if (roe.expr().size() >= 2) {
                    lo = Integer.parseInt(substituteConsts(roe.expr(0).getText(), currentEnv, List.of()));
                    hi = Integer.parseInt(substituteConsts(roe.expr(1).getText(), currentEnv, List.of()));
                } else {
                    String text = substituteConsts(roe.expr(0).getText(), currentEnv, List.of());
                    int[] bounds = parseRangeBound(text);
                    if (bounds == null) return List.of(new ActionExpansion("[" + text + "]", Map.of()));
                    lo = bounds[0]; hi = bounds[1];
                }
                List<ActionExpansion> result = new ArrayList<>();
                for (int v = lo; v <= hi; v++)
                    result.add(new ActionExpansion("[" + v + "]", Map.of(varName, v)));
                return result;
            }
            // Plain rangeOrExpr: either an explicit range (lo..hi) or a scalar/range-name.
            return expandActionRangeOrExpr(idx.rangeOrExpr());
        }

        private List<ActionExpansion> expandActionRangeOrExpr(
                FSPGrammarParser.RangeOrExprContext roe) {
            if (roe.expr().size() >= 2) {
                // Explicit range:  lo..hi
                int lo = Integer.parseInt(substituteConsts(roe.expr(0).getText(), currentEnv, List.of()));
                int hi = Integer.parseInt(substituteConsts(roe.expr(1).getText(), currentEnv, List.of()));
                List<ActionExpansion> result = new ArrayList<>();
                for (int v = lo; v <= hi; v++)
                    result.add(new ActionExpansion("[" + v + "]", Map.of()));
                return result;
            }
            // Single expr: scalar or range name.
            String text = substituteConsts(roe.expr(0).getText(), currentEnv, List.of());
            int[] bounds = parseRangeBound(text);
            if (bounds != null) {
                if (bounds[0] == bounds[1])
                    return List.of(new ActionExpansion("[" + bounds[0] + "]", Map.of()));
                List<ActionExpansion> result = new ArrayList<>();
                for (int v = bounds[0]; v <= bounds[1]; v++)
                    result.add(new ActionExpansion("[" + v + "]", Map.of()));
                return result;
            }
            return List.of(new ActionExpansion("[" + text + "]", Map.of()));
        }

        // ── composite ────────────────────────────────────────────────────────
        // ||Plant = Ejemplo.    ||Sys = (A || B).

        @Override
        public Void visitCompositeDef(FSPGrammarParser.CompositeDefContext ctx) {
            String   name = ctx.UPPER_ID().getText();
            List<LTS> components = new ArrayList<>();
            for (var item : ctx.compositeExpr().compositeItem())
                collectFromAtom(item.compositeAtom(), new HashMap<>(), components);
            composites.add(new ParallelCompositionLazy(name, List.copyOf(components)));
            return null;
        }

        // ── heuristic controller ─────────────────────────────────────────────
        // heuristic ||DirectedController = Plant~{Goal}.
        // Plant is a composite name — record as a lazy composition with no components.

        @Override
        public Void visitHeuristicDef(FSPGrammarParser.HeuristicDefContext ctx) {
            composites.add(new ParallelCompositionLazy(ctx.UPPER_ID(0).getText(), List.of()));
            return null;
        }

        // ── monolithic director ───────────────────────────────────────────────
        // monolithicDirector ||MonolithicController = Plant~{Goal}.

        @Override
        public Void visitMonolithicDef(FSPGrammarParser.MonolithicDefContext ctx) {
            composites.add(new ParallelCompositionLazy(ctx.UPPER_ID(0).getText(), List.of()));
            return null;
        }

        // ── controller ────────────────────────────────────────────────────────
        // controller ||MonolithicController = Plant~{Goal}.

        @Override
        public Void visitControllerDef(FSPGrammarParser.ControllerDefContext ctx) {
            composites.add(new ParallelCompositionLazy(ctx.UPPER_ID(0).getText(), List.of()));
            return null;
        }

        /**
         * Recursively traverse a compositeAtom, expanding forall loops and
         * instantiating each referenced process with the current loop-variable
         * bindings.  Instantiated LTS objects are appended to {@code out}.
         *
         * @param atom     the atom to process
         * @param loopEnv  current forall variable bindings (e.g. {p→0})
         * @param out      accumulator for instantiated LTS components
         */
        private void collectFromAtom(
                FSPGrammarParser.CompositeAtomContext atom,
                Map<String, Integer> loopEnv,
                List<LTS> out) {
            if (atom == null) return;

            if (atom.UPPER_ID() != null) {
                // Direct process reference:  Philosopher(p)  or  TU  or  Plant
                String processName = atom.UPPER_ID().getText();
                List<Integer> args = new ArrayList<>();
                if (atom.exprList() != null) {
                    Map<String, Integer> evalEnv = new HashMap<>(constEnv);
                    evalEnv.putAll(loopEnv);
                    for (var e : atom.exprList().expr()) {
                        String sub = substituteConsts(e.getText(), evalEnv, List.of());
                        try { args.add(Integer.parseInt(sub)); }
                        catch (NumberFormatException ignore) { /* unresolvable arg — skip */ }
                    }
                }
                LTS instance = getOrInstantiate(processName, args);
                if (instance != null) out.add(instance);

            } else if (atom.compositeExpr() != null) {
                // Grouped expression:  (A || B || C)
                for (var item : atom.compositeExpr().compositeItem())
                    collectFromAtom(item.compositeAtom(), loopEnv, out);

            } else if (atom.compositeAtom() != null) {
                // Forall loop:  forall [p:Phil] compositeAtom
                String varName = atom.LOWER_ID().getText();
                var roe = atom.rangeOrExpr();
                Map<String, Integer> evalEnv = new HashMap<>(constEnv);
                evalEnv.putAll(loopEnv);

                int lo, hi;
                if (roe.expr().size() >= 2) {
                    lo = parseWithEnv(roe.expr(0).getText(), evalEnv);
                    hi = parseWithEnv(roe.expr(1).getText(), evalEnv);
                } else {
                    String bound = substituteConsts(roe.expr(0).getText(), evalEnv, List.of());
                    int[] bounds = parseRangeBound(bound);
                    if (bounds == null) return;   // unresolvable range
                    lo = bounds[0]; hi = bounds[1];
                }

                for (int v = lo; v <= hi; v++) {
                    Map<String, Integer> inner = new HashMap<>(loopEnv);
                    inner.put(varName, v);
                    collectFromAtom(atom.compositeAtom(), inner, out);
                }
            }
        }

        /** Substitute env into text, fold arithmetic, parse as integer. */
        private int parseWithEnv(String text, Map<String, Integer> env) {
            return Integer.parseInt(substituteConsts(text, env, List.of()));
        }

        // ── set ──────────────────────────────────────────────────────────────
        // set All = {a, b, c}

        @Override
        public Void visitSetDef(FSPGrammarParser.SetDefContext ctx) {
            List<String> raw = extractSetElements(ctx.setElements());
            // Use LinkedHashSet to preserve insertion order while removing duplicates.
            List<String> deduped = new ArrayList<>(new LinkedHashSet<>(raw));
            sets.add(new SetDef(ctx.UPPER_ID().getText(), deduped));
            return null;
        }

        // ── fluent ───────────────────────────────────────────────────────────
        // fluent F1 = <initActions, termActions>
        // Represented as a 2-state LTS: "off" (initial) and "on".

        @Override
        public Void visitFluentDef(FSPGrammarParser.FluentDefContext ctx) {
            String name = ctx.UPPER_ID().getText();
            List<String> init = extractActionLabels(ctx.actionLabels(0));
            List<String> term = extractActionLabels(ctx.actionLabels(1));

            // Build the 2-state fluent LTS.
            Set<String>  allActions  = new LinkedHashSet<>(init);
            allActions.addAll(term);
            List<Transition> trans = new ArrayList<>();
            for (String a : init) {
                trans.add(new Transition("off", a, "on"));   // off → on
                trans.add(new Transition("on",  a, "on"));   // already on: self-loop
            }
            for (String t : term) {
                trans.add(new Transition("on",  t, "off"));  // on → off
                trans.add(new Transition("off", t, "off"));  // already off: self-loop
            }
            LTS fluentLTS = new LTS(
                    name, "off",
                    List.of("off", "on"),
                    List.copyOf(allActions),
                    trans,
                    true, Set.of("on"));
            fluents.add(fluentLTS);
            propertyLTSMap.put(name, fluentLTS);
            return null;
        }

        // ── assert ───────────────────────────────────────────────────────────
        // assert S1 = F1             → copy of the fluent LTS
        // assert S2 = (!FC && !FD)   → product LTS of negated fluents
        // assert S1 = (FA || FB)     → product LTS (OR accepting condition)

        @Override
        public Void visitAssertDef(FSPGrammarParser.AssertDefContext ctx) {
            String name = ctx.UPPER_ID().getText();
            LTS assertLTS = buildFltlLTS(ctx.fltlExpr(), name);
            List<String> usedFluents = extractFluentsFromFltl(ctx.fltlExpr());
            asserts.add(new LtlPropertyDef(name, usedFluents, assertLTS));
            propertyLTSMap.put(name, assertLTS);
            // If the assert is a simple alias (e.g. "assert A = F"), remember the mapping.
            String simple = extractSimpleFltlRef(ctx.fltlExpr());
            if (simple != null) simpleAssertMap.put(name, simple);
            return null;
        }

        /** Collect all UPPER_ID references (fluent/property names) used in an FLTL expression. */
        private List<String> extractFluentsFromFltl(FSPGrammarParser.FltlExprContext ctx) {
            Set<String> names = new LinkedHashSet<>();
            collectUpperIds(ctx, names);
            return new ArrayList<>(names);
        }

        private void collectUpperIds(org.antlr.v4.runtime.tree.ParseTree tree, Set<String> names) {
            if (tree instanceof org.antlr.v4.runtime.tree.TerminalNode) {
                org.antlr.v4.runtime.Token tok =
                        ((org.antlr.v4.runtime.tree.TerminalNode) tree).getSymbol();
                if (tok.getType() == FSPGrammarLexer.UPPER_ID) {
                    names.add(tok.getText());
                }
            } else {
                for (int i = 0; i < tree.getChildCount(); i++) {
                    collectUpperIds(tree.getChild(i), names);
                }
            }
        }

        /**
         * Returns the single UPPER_ID name if the fltlExpr is just a bare identifier,
         * e.g. "assert A = F" → "F".  Returns null for complex expressions.
         */
        private String extractSimpleFltlRef(FSPGrammarParser.FltlExprContext ctx) {
            var orCtx = ctx.fltlOrExpr();
            if (orCtx.fltlBinExpr().size() != 1) return null;
            var binCtx = orCtx.fltlBinExpr(0);
            if (binCtx.fltlAndExpr().size() != 1) return null;
            var andCtx = binCtx.fltlAndExpr(0);
            if (andCtx.fltlUnaryExpr().size() != 1) return null;
            var unaryCtx = andCtx.fltlUnaryExpr(0);
            if (unaryCtx.fltlBaseExpr() == null) return null;  // has unary op
            var baseCtx = unaryCtx.fltlBaseExpr();
            return baseCtx.UPPER_ID() != null ? baseCtx.UPPER_ID().getText() : null;
        }

        // ── FLTL → LTS conversion ─────────────────────────────────────────────

        /** Build an LTS from an FLTL expression. The resulting LTS name is {@code name}. */
        private LTS buildFltlLTS(FSPGrammarParser.FltlExprContext ctx, String name) {
            LTS result = evalFltlOr(ctx.fltlOrExpr());
            // Rename to the declared assert name if provided.
            if (name != null && !name.equals(result.name()))
                result = new LTS(name, result.initialState(), result.states(),
                        result.actions(), result.transitions(), false, result.acceptingStates());
            return result;
        }

        private LTS evalFltlOr(FSPGrammarParser.FltlOrExprContext ctx) {
            List<FSPGrammarParser.FltlBinExprContext> bins = ctx.fltlBinExpr();
            LTS result = evalFltlBin(bins.get(0));
            for (int i = 1; i < bins.size(); i++)
                result = ltsOr(result, evalFltlBin(bins.get(i)));
            return result;
        }

        private LTS evalFltlBin(FSPGrammarParser.FltlBinExprContext ctx) {
            // Handles -> and <-> (A->B = !A||B, A<->B = (A&&B)||(!A&&!B)).
            // For now fall back to the first operand if more complex.
            List<FSPGrammarParser.FltlAndExprContext> ands = ctx.fltlAndExpr();
            LTS result = evalFltlAnd(ands.get(0));
            for (int i = 1; i < ands.size(); i++) {
                LTS right = evalFltlAnd(ands.get(i));
                String op = ctx.getChild(2 * i - 1).getText();
                if (op.equals("->"))       result = ltsOr(ltsNeg(result), right);       // A->B = !A||B
                else if (op.equals("<->")) result = ltsOr(ltsAnd(result, right),          // A<->B
                                                          ltsAnd(ltsNeg(result), ltsNeg(right)));
            }
            return result;
        }

        private LTS evalFltlAnd(FSPGrammarParser.FltlAndExprContext ctx) {
            List<FSPGrammarParser.FltlUnaryExprContext> unaries = ctx.fltlUnaryExpr();
            LTS result = evalFltlUnary(unaries.get(0));
            for (int i = 1; i < unaries.size(); i++)
                result = ltsAnd(result, evalFltlUnary(unaries.get(i)));
            return result;
        }

        private LTS evalFltlUnary(FSPGrammarParser.FltlUnaryExprContext ctx) {
            if (ctx.fltlBaseExpr() != null) return evalFltlBase(ctx.fltlBaseExpr());
            LTS inner = evalFltlUnary(ctx.fltlUnaryExpr());
            String op = ctx.getChild(0).getText();
            if (op.equals("!")) return ltsNeg(inner);
            // [] (always) and <> (eventually) are temporal — pass through for now.
            return inner;
        }

        private LTS evalFltlBase(FSPGrammarParser.FltlBaseExprContext ctx) {
            if (ctx.UPPER_ID() != null) {
                String refName = ctx.UPPER_ID().getText();
                LTS ref = propertyLTSMap.get(refName);
                if (ref != null) return ref;
                // Unknown reference — return a stub LTS (should not happen in well-formed FSP).
                return new LTS(refName, "off", List.of("off", "on"), List.of(),
                        List.of(), false, Set.of("on"));
            }
            if (ctx.fltlExpr() != null) return buildFltlLTS(ctx.fltlExpr(), null);
            // LOWER_ID → treat as a single-action inline fluent: on when this action fires,
            // off for all other actions in the known alphabet.
            String action = ctx.getStart().getText();
            return buildActionFluent(action);
        }

        /**
         * Build a 2-state inline fluent for an action label appearing in an FLTL expression.
         * Equivalent to {@code fluent <action, All\{action}>}: on when {@code action} fires,
         * off for every other action in the known alphabet (collected from defined sets and fluents).
         */
        private LTS buildActionFluent(String action) {
            // Collect the full alphabet from set definitions and from already-defined fluents.
            Set<String> alphabet = new LinkedHashSet<>();
            for (SetDef s : sets)   alphabet.addAll(s.elements());
            for (LTS    f : fluents) alphabet.addAll(f.actions());
            alphabet.add(action); // guarantee the action itself is present

            List<Transition> trans = new ArrayList<>();
            // The action initiates the fluent (off→on) and re-initiates (on→on).
            trans.add(new Transition("off", action, "on"));
            trans.add(new Transition("on",  action, "on"));
            // Every other known action terminates the fluent.
            for (String a : alphabet) {
                if (!a.equals(action)) {
                    trans.add(new Transition("off", a, "off"));
                    trans.add(new Transition("on",  a, "off"));
                }
            }
            // isFluent=true so the synthesiser self-loops on actions outside our alphabet
            // (self-loop in "off" is harmless; "on" should not self-loop but we have explicit
            // transitions for all known actions, so unknown ones would be rare in practice).
            return new LTS(action, "off", List.of("off", "on"),
                    List.copyOf(alphabet), trans, true, Set.of("on"));
        }

        // ── LTS boolean combinators ───────────────────────────────────────────

        /**
         * Negate a property LTS: flip accepting ↔ non-accepting states.
         * The transitions and alphabet are unchanged; only {@code acceptingStates} changes.
         */
        private LTS ltsNeg(LTS lts) {
            Set<String> negAccepting = lts.states().stream()
                    .filter(s -> !lts.acceptingStates().contains(s))
                    .collect(Collectors.toSet());
            return new LTS(lts.name(), lts.initialState(), lts.states(),
                    lts.actions(), lts.transitions(), false, negAccepting);
        }

        /** AND product: accepting when BOTH component states are accepting. */
        private LTS ltsAnd(LTS a, LTS b) {
            return ltsProduct(a, b, (sa, sb) ->
                    a.acceptingStates().contains(sa) && b.acceptingStates().contains(sb));
        }

        /** OR product: accepting when AT LEAST ONE component state is accepting. */
        private LTS ltsOr(LTS a, LTS b) {
            return ltsProduct(a, b, (sa, sb) ->
                    a.acceptingStates().contains(sa) || b.acceptingStates().contains(sb));
        }

        /**
         * Synchronous parallel composition of two property LTSes.
         * <p>
         * State names are joined with "|" (e.g. "on|off").
         * For each action, if a component has no explicit transition from a given state,
         * it self-loops (the action is not in its alphabet / no entry in transition map).
         * </p>
         */
        private LTS ltsProduct(LTS a, LTS b,
                               java.util.function.BiPredicate<String,String> isAccepting) {
            // Build per-state transition maps: state → action → successor.
            Map<String, Map<String, String>> aMap = buildTransMap(a);
            Map<String, Map<String, String>> bMap = buildTransMap(b);

            Set<String> allActions = new LinkedHashSet<>(a.actions());
            allActions.addAll(b.actions());

            List<String> productStates = new ArrayList<>();
            List<Transition> productTrans = new ArrayList<>();
            Set<String> accepting       = new LinkedHashSet<>();

            for (String sa : a.states()) {
                for (String sb : b.states()) {
                    String from = sa + "_" + sb;
                    productStates.add(from);
                    if (isAccepting.test(sa, sb)) accepting.add(from);

                    for (String act : allActions) {
                        // Self-loop if no explicit outgoing transition.
                        String sa2 = aMap.getOrDefault(sa, Map.of()).getOrDefault(act, sa);
                        String sb2 = bMap.getOrDefault(sb, Map.of()).getOrDefault(act, sb);
                        productTrans.add(new Transition(from, act, sa2 + "_" + sb2));
                    }
                }
            }

            String initialState = a.initialState() + "_" + b.initialState();
            return new LTS(
                    a.name() + "_" + b.name(),
                    initialState,
                    productStates,
                    List.copyOf(allActions),
                    productTrans,
                    false, accepting);
        }

        /** Build a per-state, per-action transition map from an LTS (for product computation). */
        private Map<String, Map<String, String>> buildTransMap(LTS lts) {
            Map<String, Map<String, String>> map = new HashMap<>();
            for (Transition t : lts.transitions())
                map.computeIfAbsent(t.from(), k -> new LinkedHashMap<>())
                   .putIfAbsent(t.action(), t.to());  // first wins (deterministic)
            return map;
        }

        // ── ltl_property ─────────────────────────────────────────────────────
        // ltl_property SAF = []NB

        @Override
        public Void visitLtlPropertyDef(FSPGrammarParser.LtlPropertyDefContext ctx) {
            // ltl_property P = expr
            //
            // For safety use (safety = {P}), the controller must ensure expr holds always.
            // We build the LTS for `expr` (temporal operators [] / <> are stripped, they
            // only add liveness/safety intent beyond what the LTS already captures).
            // The resulting acceptingStates = states where expr is TRUE = SAFE states.
            // States NOT in acceptingStates are the ILLEGAL states for this property.
            String name = ctx.UPPER_ID().getText();
            LTS safetyLTS = buildFltlLTS(ctx.fltlExpr(), name);
            List<String> usedFluents = extractFluentsFromFltl(ctx.fltlExpr());
            ltlProperties.add(new LtlPropertyDef(name, usedFluents, safetyLTS));
            propertyLTSMap.put(name, safetyLTS);
            return null;
        }

        // ── controllerSpec ────────────────────────────────────────────────────
        // controllerSpec Goal = { liveness = {S1}, controllable = {a}, ... }

        @Override
        public Void visitControllerSpecDef(FSPGrammarParser.ControllerSpecDefContext ctx) {
            String name = ctx.UPPER_ID().getText();

            List<String> liveness     = new ArrayList<>();
            List<String> safety       = new ArrayList<>();
            List<String> assumption   = new ArrayList<>();
            List<String> controllable = new ArrayList<>();
            List<String> marking      = new ArrayList<>();
            boolean nonblocking       = false;

            for (FSPGrammarParser.SpecFieldContext f : ctx.specField()) {
                if      (f.LIVENESS()     != null) liveness.addAll(extractUpperIdList(f.upperIdList()));
                else if (f.SAFETY()       != null) safety.addAll(extractUpperIdList(f.upperIdList()));
                else if (f.ASSUMPTION()   != null) assumption.addAll(extractUpperIdList(f.upperIdList()));
                else if (f.CONTROLLABLE() != null) controllable.addAll(extractExtSetElements(f.extSetElements()));
                else if (f.MARKING()      != null) marking.addAll(extractExtSetElements(f.extSetElements()));
                else if (f.NONBLOCKING()  != null) nonblocking = true;
            }

            controllerSpecs.add(new ControllerSpecDef(
                    name,
                    List.copyOf(liveness),
                    List.copyOf(safety),
                    List.copyOf(assumption),
                    List.copyOf(controllable),
                    List.copyOf(marking),
                    nonblocking
            ));
            return null;
        }

        // ── helpers ───────────────────────────────────────────────────────────

        private List<String> extractUpperIdList(FSPGrammarParser.UpperIdListContext ctx) {
            if (ctx == null) return List.of();
            return ctx.UPPER_ID().stream()
                    .map(t -> t.getText())
                    .collect(Collectors.toList());
        }

        /**
         * Expand all elements of a setElements node to concrete action strings.
         * Handles dotted/indexed labels (via expandLabelBase) and set references
         * (UPPER_ID, optionally with set-difference).
         */
        private List<String> extractSetElements(FSPGrammarParser.SetElementsContext ctx) {
            if (ctx == null) return List.of();
            List<String> result = new ArrayList<>();
            for (var e : ctx.setElement()) {
                if (e.labelBase() != null) {
                    result.addAll(expandLabelBase(e.labelBase()));
                } else {
                    // UPPER_ID setDiff?  — named set reference, optionally subtracted
                    List<String> base = resolveSetByName(e.UPPER_ID().getText());
                    if (e.setDiff() != null) {
                        Set<String> toRemove = new HashSet<>(extractSetElements(e.setDiff().setElements()));
                        base = base.stream().filter(s -> !toRemove.contains(s)).collect(Collectors.toList());
                    }
                    result.addAll(base);
                }
            }
            return result;
        }

        /** Look up a named set and return its expanded elements; falls back to [name] if unknown. */
        private List<String> resolveSetByName(String name) {
            return sets.stream()
                    .filter(s -> s.name().equals(name))
                    .findFirst()
                    .map(SetDef::elements)
                    .orElse(List.of(name));
        }

        private List<String> extractExtSetElements(FSPGrammarParser.ExtSetElementsContext ctx) {
            if (ctx == null) return List.of();
            List<String> result = new ArrayList<>();
            for (var e : ctx.extSetElement()) {
                if (e.labelBase() != null) {
                    result.addAll(expandLabelBase(e.labelBase()));
                } else if (e.UPPER_ID() != null) {
                    // Resolve named set; fall back to the name itself if not found
                    List<String> resolved = resolveSetByName(e.UPPER_ID().getText());
                    if (!resolved.isEmpty()) result.addAll(resolved);
                    else result.add(e.UPPER_ID().getText());
                } else if (e.INT() != null) {
                    result.add(e.INT().getText());
                }
            }
            return result;
        }

        /**
         * Expand a labelBase to all concrete action strings.
         * Handles dotted segments (eat.all) and range indices (take[Phil][Phil]).
         */
        private List<String> expandLabelBase(FSPGrammarParser.LabelBaseContext lb) {
            List<List<String>> segExpansions = new ArrayList<>();
            for (var seg : lb.labelSegment())
                segExpansions.add(expandLabelSegment(seg));
            // Cartesian product of segments, joined with "."
            List<String> acc = List.of("");
            for (List<String> part : segExpansions) {
                List<String> next = new ArrayList<>();
                for (String prefix : acc)
                    for (String s : part)
                        next.add(prefix.isEmpty() ? s : prefix + "." + s);
                acc = next;
            }
            return acc;
        }

        private List<String> expandLabelSegment(FSPGrammarParser.LabelSegmentContext seg) {
            String base = seg.LOWER_ID().getText();
            if (seg.labelIndex().isEmpty()) return List.of(base);
            // Cartesian product of all index expansions
            List<String> indexPart = List.of("");
            for (var idx : seg.labelIndex()) {
                List<String> expanded = expandLabelIndex(idx);
                List<String> next = new ArrayList<>();
                for (String prefix : indexPart)
                    for (String s : expanded)
                        next.add(prefix + s);
                indexPart = next;
            }
            String base0 = base;
            return indexPart.stream().map(i -> base0 + i).collect(Collectors.toList());
        }

        /** Expand one label index bracket to a list of "[value]" strings. */
        private List<String> expandLabelIndex(FSPGrammarParser.LabelIndexContext idx) {
            if (idx.anyId() != null) {
                // Variable binding:  b:Area  — expand the range, ignore var name
                var roe = idx.rangeOrExpr();
                return expandRangeOrExpr(roe);
            }
            return expandRangeOrExpr(idx.rangeOrExpr());
        }

        private List<String> expandRangeOrExpr(FSPGrammarParser.RangeOrExprContext roe) {
            if (roe.expr().size() >= 2) {
                // Explicit range:  lo..hi
                String loText = substituteConsts(roe.expr(0).getText(), constEnv, List.of());
                String hiText = substituteConsts(roe.expr(1).getText(), constEnv, List.of());
                try {
                    int lo = Integer.parseInt(loText), hi = Integer.parseInt(hiText);
                    List<String> r = new ArrayList<>();
                    for (int v = lo; v <= hi; v++) r.add("[" + v + "]");
                    return r;
                } catch (NumberFormatException e) {
                    return List.of("[" + loText + ".." + hiText + "]");
                }
            }
            // Single expr: a scalar or a range name
            String text = substituteConsts(roe.expr(0).getText(), constEnv, List.of());
            int[] bounds = parseRangeBound(text);
            if (bounds != null) {
                if (bounds[0] == bounds[1]) return List.of("[" + bounds[0] + "]");
                List<String> r = new ArrayList<>();
                for (int v = bounds[0]; v <= bounds[1]; v++) r.add("[" + v + "]");
                return r;
            }
            return List.of("[" + text + "]");
        }

        /**
         * Flatten an actionLabels node to a list of concrete action strings.
         *  - labelBase         → expand to all concrete labels (dotted/indexed)
         *  - UPPER_ID setDiff? → resolve named set, apply optional set-difference
         *  - { setElements }   → expand all elements
         */
        private List<String> extractActionLabels(FSPGrammarParser.ActionLabelsContext ctx) {
            if (ctx == null) return List.of();
            if (ctx.labelBase() != null) return expandLabelBase(ctx.labelBase());
            if (ctx.UPPER_ID() != null) {
                List<String> base = resolveSetByName(ctx.UPPER_ID().getText());
                if (ctx.setDiff() != null) {
                    Set<String> toRemove = new HashSet<>(extractSetElements(ctx.setDiff().setElements()));
                    base = base.stream().filter(s -> !toRemove.contains(s)).collect(Collectors.toList());
                }
                return base;
            }
            return extractSetElements(ctx.setElements());
        }
    }
}
