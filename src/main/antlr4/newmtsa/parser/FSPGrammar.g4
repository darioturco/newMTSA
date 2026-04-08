grammar FSPGrammar;

// ═══════════════════════════════════════════════════════════════
//  TOP LEVEL
// ═══════════════════════════════════════════════════════════════

fspDescription
    : definition* EOF
    ;

definition
    : processDef
    | heuristicDef          // keyword-prefixed; before compositeDef
    | monolithicDef         // keyword-prefixed
    | controllerDef         // keyword-prefixed
    | compositeDef
    | constDef
    | rangeDef
    | defMacroDef
    | setDef
    | fluentDef
    | assertDef
    | ltlPropertyDef
    | controllerSpecDef
    ;

// ═══════════════════════════════════════════════════════════════
//  CONST / RANGE / DEF  (parametric FSP)
//
//  const N = 2
//  range Phil = 0..N-1
//  def LeftP(p) = p
//  def LeftF(f) = f == 0 ? N-1 : f-1
// ═══════════════════════════════════════════════════════════════

constDef
    : CONST UPPER_ID '=' expr
    ;

rangeDef
    : RANGE UPPER_ID '=' rangeOrExpr
    ;

defMacroDef
    : DEF anyId '(' anyId (',' anyId)* ')' '=' expr
    ;

// ═══════════════════════════════════════════════════════════════
//  PROCESS DEFINITION
//
//  Ejemplo = A0,
//  A0 = (a -> A1),
//  A1 = (b -> A2 | c -> A0).
//
//  Parametric form:
//  Machine(Id=0) = Working[0],
//  Working[wl:0..Workload] = (when (wl < Workload) get[Id] -> Working[wl+1] | ...).
// ═══════════════════════════════════════════════════════════════

processDef
    : UPPER_ID ('(' paramDefList ')')? '=' localProcess (',' localProcessDef)* '.'
    ;

paramDefList
    : paramDef (',' paramDef)*
    ;

paramDef
    : anyId ('=' expr)?
    ;

localProcessDef
    : UPPER_ID ('[' indexSpec ']')* '=' localProcess
    ;

// indexSpec covers both:
//   [s:1..Steps]   — range variable declaration
//   [0]            — concrete index expression
indexSpec
    : anyId ':' rangeOrExpr
    | expr
    ;

localProcess
    : '(' choice ')' alphabetExt?
    | IF '(' expr ')' THEN localProcess ELSE localProcess
    | UPPER_ID ('[' expr ']')*
    ;

// Alphabet extension:  +{return[0..Machines]}
alphabetExt
    : '+' '{' extSetElements '}'
    ;

choice
    : guardedPrefix ('|' guardedPrefix)*
    ;

// Optional 'when (cond)' guard before the action prefix
guardedPrefix
    : (WHEN '(' expr ')')? actionPrefix
    ;

// PrefixActions -> LocalProcess
// Chaining: a -> b -> State  (a -> b is the prefix)
actionPrefix
    : prefixActions '->' localProcess
    ;

// Only lowercase labels (possibly dotted / indexed) are valid action labels
// in process bodies.  This keeps the lowercase/uppercase disambiguation that
// allows the parser to know when the chain ends and a state name follows.
prefixActions
    : processActionLabel ('->' processActionLabel)*
    ;

processActionLabel
    : labelBase
    | '{' processActionLabel (',' processActionLabel)* '}'
    ;

// Index inside a label bracket: either a variable binding (b:Area) or a range/expr.
labelIndex
    : anyId ':' rangeOrExpr   // variable binding:  b:Area
    | rangeOrExpr              // range or scalar:   0..Cats-1  |  a+1
    ;

// One segment of a dotted label, with optional indexed parameters.
//   cat[0..Cats-1]   move[b:Area]   eat   think[Pid]
labelSegment
    : LOWER_ID ('[' labelIndex ']')*
    ;

// Dotted action labels with optional indexed segments:
//   eat.all   step.next   cat[0..Cats-1].move[b:Area]
labelBase
    : labelSegment ('.' labelSegment)*
    ;

// ═══════════════════════════════════════════════════════════════
//  COMPOSITE DEFINITION
//
//  ||Plant = Ejemplo.
//  ||Sys   = (DirectedController || Plant).
//  ||Plant = (forall [p:Phil] (Philosopher(p) || Fork(p) || Monitor(p))).
// ═══════════════════════════════════════════════════════════════

compositeDef
    : '||' UPPER_ID '=' compositeExpr '.'
    ;

compositeExpr
    : compositeItem ('||' compositeItem)*
    ;

compositeItem
    : compositeAtom alphabetExt?
    ;

compositeAtom
    : UPPER_ID ('(' exprList ')')?
    | '(' compositeExpr ')'
    | FORALL '[' LOWER_ID ':' rangeOrExpr ']' compositeAtom
    ;

// ═══════════════════════════════════════════════════════════════
//  HEURISTIC CONTROLLER
//
//  heuristic ||DirectedController = Plant~{Goal}.
// ═══════════════════════════════════════════════════════════════

heuristicDef
    : HEURISTIC '||' UPPER_ID '=' UPPER_ID '~' '{' UPPER_ID '}' '.'
    ;

// ═══════════════════════════════════════════════════════════════
//  MONOLITHIC DIRECTOR
//
//  monolithicDirector ||MonolithicController = Plant~{Goal}.
// ═══════════════════════════════════════════════════════════════

monolithicDef
    : MONOLITHIC_DIRECTOR '||' UPPER_ID '=' UPPER_ID '~' '{' UPPER_ID '}' '.'
    ;

// ═══════════════════════════════════════════════════════════════
//  CONTROLLER  (MTSA — keyword 'controller', different from controllerSpec)
//
//  controller ||MonolithicController = Plant~{Goal}.
// ═══════════════════════════════════════════════════════════════

controllerDef
    : CONTROLLER_LC '||' UPPER_ID '=' UPPER_ID '~' '{' UPPER_ID '}' '.'
    ;

// ═══════════════════════════════════════════════════════════════
//  SET DEFINITION
//
//  set All = {a, b, c}
// ═══════════════════════════════════════════════════════════════

setDef
    : SET UPPER_ID '=' '{' setElements '}'
    ;

setElements
    : setElement (',' setElement)*
    ;

setElement
    : labelBase
    | UPPER_ID setDiff?
    ;

setDiff
    : '\\' '{' setElements '}'
    ;

// ═══════════════════════════════════════════════════════════════
//  EXTENDED SET ELEMENTS
//  Used in controllerSpec controllable/marking fields, which can contain
//  indexed labels and range expressions:
//    {take[Phil][Phil]}   {get[0..Machines]}   {eat.all}
// ═══════════════════════════════════════════════════════════════

extSetElements
    : extSetElement (',' extSetElement)*
    ;

extSetElement
    : labelBase
    | UPPER_ID setDiff?
    | INT
    ;

// ═══════════════════════════════════════════════════════════════
//  FLUENT DEFINITION
//
//  fluent F1 = <a, b>
//  fluent F2 = <b, All\{b}>
//  fluent F3 = <{d,g,h}, {e,f}>
// ═══════════════════════════════════════════════════════════════

fluentDef
    : FLUENT UPPER_ID '=' '<' actionLabels ',' actionLabels '>'
    ;

actionLabels
    : LOWER_ID
    | '{' setElements '}'
    | UPPER_ID setDiff?
    ;

// ═══════════════════════════════════════════════════════════════
//  ASSERT DEFINITION
//
//  assert S1 = F1
//  assert Check = ([]<>S1 && []<>S2)
// ═══════════════════════════════════════════════════════════════

assertDef
    : ASSERT UPPER_ID '=' fltlExpr
    ;

// ═══════════════════════════════════════════════════════════════
//  LTL_PROPERTY DEFINITION
//
//  ltl_property SAF = []NB
// ═══════════════════════════════════════════════════════════════

ltlPropertyDef
    : LTL_PROPERTY UPPER_ID '=' fltlExpr
    ;

// ═══════════════════════════════════════════════════════════════
//  CONTROLLER SPEC (MTSA extension)
//
//  controllerSpec Goal = {
//      liveness     = {S1, S2}
//      safety       = {SAF}
//      assumption   = {A1, A2}
//      controllable = {a, b}
//      marking      = {u55}
//      nonblocking
//  }
// ═══════════════════════════════════════════════════════════════

controllerSpecDef
    : CONTROLLER_SPEC UPPER_ID '=' '{' specField* '}'
    ;

specField
    : LIVENESS     '=' '{' upperIdList '}'
    | SAFETY       '=' '{' upperIdList '}'
    | ASSUMPTION   '=' '{' upperIdList '}'
    | CONTROLLABLE '=' '{' extSetElements '}'
    | MARKING      '=' '{' extSetElements '}'
    | NONBLOCKING
    ;

upperIdList
    : UPPER_ID (',' UPPER_ID)*
    ;

// ═══════════════════════════════════════════════════════════════
//  EXPRESSIONS  (arithmetic, comparison, ternary)
//
//  Used in: const, range, def, process params, when guards,
//           indexed action labels, indexed state references.
//
//  Precedence (lowest → highest):
//    ternary       expr ? expr : expr
//    comparison    == != < > <= >=
//    additive      + -
//    multiplicative * / %
//    unary         - (negation)
//    primary       ( expr )  |  id ( args )  |  id  |  INT
// ═══════════════════════════════════════════════════════════════

exprList
    : expr (',' expr)*
    ;

// Range-or-single-expr:  0..N-1  or  Phil  or  0
rangeOrExpr
    : expr (DOTDOT expr)?
    ;

anyId
    : LOWER_ID
    | UPPER_ID
    ;

expr
    : <assoc=right> expr '?' expr ':' expr         // ternary
    | expr '||' expr                               // logical or
    | expr '&&' expr                               // logical and
    | expr ('=='|'!='|'<'|'>'|'<='|'>=') expr      // comparison
    | expr ('*'|'/'|'%'|'\\') expr                   // multiplicative (\ = integer division)
    | expr ('+'|'-') expr                           // additive
    | '-' expr                                      // unary minus
    | '(' expr ')'                                  // grouping
    | anyId ('(' exprList ')')?                     // variable or macro call
    | INT                                           // integer literal
    ;

// ═══════════════════════════════════════════════════════════════
//  FLTL EXPRESSIONS
//
//  Precedence (lowest → highest):
//    ||          or
//    -> <->      implication / biconditional
//    &&          and
//    ! [] <>     unary not / always / eventually
//    atom        UPPER_ID | LOWER_ID | ( expr )
// ═══════════════════════════════════════════════════════════════

fltlExpr
    : fltlOrExpr
    ;

fltlOrExpr
    : fltlBinExpr ('||' fltlBinExpr)*
    ;

fltlBinExpr
    : fltlAndExpr (('->' | '<->') fltlAndExpr)*
    ;

fltlAndExpr
    : fltlUnaryExpr ('&&' fltlUnaryExpr)*
    ;

fltlUnaryExpr
    : '!'        fltlUnaryExpr
    | ALWAYS     fltlUnaryExpr     // []
    | EVENTUALLY fltlUnaryExpr     // <>
    | fltlBaseExpr
    ;

fltlBaseExpr
    : UPPER_ID              // fluent name or assert reference
    | LOWER_ID              // action label used as proposition
    | '(' fltlExpr ')'
    ;

// ═══════════════════════════════════════════════════════════════
//  LEXER  ── keywords must appear before the catch-all ID rules
// ═══════════════════════════════════════════════════════════════

// MTSA / FSP keywords
FLUENT               : 'fluent' ;
ASSERT               : 'assert' ;
SET                  : 'set' ;
HEURISTIC            : 'heuristic' ;
MONOLITHIC_DIRECTOR  : 'monolithicDirector' ;
CONTROLLER_LC        : 'controller' ;
LTL_PROPERTY         : 'ltl_property' ;
CONTROLLER_SPEC      : 'controllerSpec' ;
CONST                : 'const' ;
RANGE                : 'range' ;
DEF                  : 'def' ;
WHEN                 : 'when' ;
FORALL               : 'forall' ;
IF                   : 'if' ;
THEN                 : 'then' ;
ELSE                 : 'else' ;

// controllerSpec field names
LIVENESS             : 'liveness' ;
SAFETY               : 'safety' ;
CONTROLLABLE         : 'controllable' ;
MARKING              : 'marking' ;
NONBLOCKING          : 'nonblocking' ;
ASSUMPTION           : 'assumption' ;

// Temporal-logic operators (before '<' single-char so maximal munch applies)
ALWAYS               : '[]' ;
EVENTUALLY           : '<>' ;

// Range separator — two dots — must appear before any single-dot token
DOTDOT               : '..' ;

// Integer literals
INT                  : [0-9]+ ;

// Identifiers
UPPER_ID : [A-Z][a-zA-Z0-9_]* ;
LOWER_ID : [a-z_][a-zA-Z0-9_]* ;

// Ignored
WS           : [ \t\r\n]+          -> skip ;
LINE_COMMENT : '//' ~[\r\n]*       -> skip ;
BLOCK_COMMENT: '/*' .*? '*/'       -> skip ;
