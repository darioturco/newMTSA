package newmtsa.parser;

import newmtsa.parser.ast.*;
import org.antlr.v4.runtime.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class FSPParser {

    public static FSPModel parse(Path file) throws IOException {
        CharStream chars = CharStreams.fromString(Files.readString(file));

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

        return new FSPModel(
                List.copyOf(visitor.processes),
                List.copyOf(visitor.composites),
                List.copyOf(visitor.fluents),
                List.copyOf(visitor.asserts),
                List.copyOf(visitor.ltlProperties),
                List.copyOf(visitor.sets),
                List.copyOf(visitor.controllerSpecs),
                List.copyOf(errors)
        );
    }

    // ─── Visitor ──────────────────────────────────────────────────────────────
    // Walks the parse tree and collects top-level definitions into plain lists.
    // Each visitX method records the definition it cares about and returns null
    // (no deeper recursion needed — we only need top-level names and fields).

    private static final class Visitor extends FSPGrammarBaseVisitor<Void> {

        final List<ProcessDef>        processes       = new ArrayList<>();
        final List<CompositeDef>      composites      = new ArrayList<>();
        final List<FluentDef>         fluents         = new ArrayList<>();
        final List<AssertDef>         asserts         = new ArrayList<>();
        final List<LtlPropertyDef>    ltlProperties   = new ArrayList<>();
        final List<SetDef>            sets            = new ArrayList<>();
        final List<ControllerSpecDef> controllerSpecs = new ArrayList<>();

        // ── process ──────────────────────────────────────────────────────────
        // Ejemplo = A0, A0 = (a -> A1), ...

        @Override
        public Void visitProcessDef(FSPGrammarParser.ProcessDefContext ctx) {
            processes.add(new ProcessDef(ctx.UPPER_ID().getText()));
            return null;
        }

        // ── composite ────────────────────────────────────────────────────────
        // ||Plant = Ejemplo.    ||Sys = (A || B).

        @Override
        public Void visitCompositeDef(FSPGrammarParser.CompositeDefContext ctx) {
            composites.add(new CompositeDef(ctx.UPPER_ID().getText()));
            return null;
        }

        // ── heuristic controller ─────────────────────────────────────────────
        // heuristic ||DirectedController = Plant~{Goal}.
        // Treated as a composite — first UPPER_ID is the controller name.

        @Override
        public Void visitHeuristicDef(FSPGrammarParser.HeuristicDefContext ctx) {
            composites.add(new CompositeDef(ctx.UPPER_ID(0).getText()));
            return null;
        }

        // ── monolithic director ───────────────────────────────────────────────
        // monolithicDirector ||MonolithicController = Plant~{Goal}.
        // Treated as a composite — first UPPER_ID is the controller name.

        @Override
        public Void visitMonolithicDef(FSPGrammarParser.MonolithicDefContext ctx) {
            composites.add(new CompositeDef(ctx.UPPER_ID(0).getText()));
            return null;
        }

        // ── set ──────────────────────────────────────────────────────────────
        // set All = {a, b, c}

        @Override
        public Void visitSetDef(FSPGrammarParser.SetDefContext ctx) {
            sets.add(new SetDef(
                    ctx.UPPER_ID().getText(),
                    extractSetElements(ctx.setElements())
            ));
            return null;
        }

        // ── fluent ───────────────────────────────────────────────────────────
        // fluent F1 = <a, All\{b}>

        @Override
        public Void visitFluentDef(FSPGrammarParser.FluentDefContext ctx) {
            fluents.add(new FluentDef(
                    ctx.UPPER_ID().getText(),
                    extractActionLabels(ctx.actionLabels(0)),
                    extractActionLabels(ctx.actionLabels(1))
            ));
            return null;
        }

        // ── assert ───────────────────────────────────────────────────────────
        // assert S1 = ([]<>F1 && []<>F2)

        @Override
        public Void visitAssertDef(FSPGrammarParser.AssertDefContext ctx) {
            asserts.add(new AssertDef(ctx.UPPER_ID().getText()));
            return null;
        }

        // ── ltl_property ─────────────────────────────────────────────────────
        // ltl_property SAF = []NB

        @Override
        public Void visitLtlPropertyDef(FSPGrammarParser.LtlPropertyDefContext ctx) {
            ltlProperties.add(new LtlPropertyDef(ctx.UPPER_ID().getText()));
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

        /** Returns the leading token text of each set element (the action name or set name). */
        private List<String> extractSetElements(FSPGrammarParser.SetElementsContext ctx) {
            if (ctx == null) return List.of();
            return ctx.setElement().stream()
                    .map(e -> e.getStart().getText())
                    .collect(Collectors.toList());
        }

        private List<String> extractExtSetElements(FSPGrammarParser.ExtSetElementsContext ctx) {
            if (ctx == null) return List.of();
            return ctx.extSetElement().stream()
                    .map(e -> e.getStart().getText())
                    .collect(Collectors.toList());
        }

        /**
         * Flatten an actionLabels node to a list of action-label strings.
         *  - LOWER_ID          → ["a"]
         *  - UPPER_ID setDiff? → ["All"]   (set reference, resolved later)
         *  - { setElements }   → ["a","b","c"]
         */
        private List<String> extractActionLabels(FSPGrammarParser.ActionLabelsContext ctx) {
            if (ctx == null) return List.of();
            if (ctx.LOWER_ID() != null) return List.of(ctx.LOWER_ID().getText());
            if (ctx.UPPER_ID() != null) return List.of(ctx.UPPER_ID().getText());
            return extractSetElements(ctx.setElements());
        }
    }
}
