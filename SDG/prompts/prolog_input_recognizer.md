You are an expert Prolog programmer specializing in pattern recognition and constraint logic programming.

Your task is to write a Prolog predicate `valid_input(Grid)` that recognizes valid input grids for an ARC puzzle.

## Grid Format

Grids are represented as lists of lists in Prolog:
- A 2x3 grid: `[[1,2,3],[4,5,6]]` (2 rows, 3 columns)
- Each cell contains a color value from 0-9
- Colors: 0=BLACK, 1=BLUE, 2=RED, 3=GREEN, 4=YELLOW, 5=GRAY, 6=MAGENTA, 7=ORANGE, 8=SKY, 9=BROWN

## Train Input Examples

The following are valid input grids from this puzzle. Your predicate MUST accept ALL of these:

{TRAIN_INPUTS}

## Your Analysis Process

Before writing code, analyze the input grids inside <analysis> tags:

1. **Dimensions**: What are the grid dimensions? Are they fixed or variable?
2. **Colors used**: Which colors appear? Are some colors required/forbidden?
3. **Spatial patterns**: Look for:
   - Symmetry (horizontal, vertical, rotational)
   - Repeating patterns or tiles
   - Objects/shapes and their properties
   - Relationships between cells (adjacency, alignment)
4. **Constraints**: What rules must a valid input satisfy?

## Requirements

Your predicate must:
1. **Accept ALL train examples** - This is mandatory
2. **Be SPECIFIC** - Capture actual structural constraints visible in the examples
3. **NOT be trivial** - Do NOT write overly general predicates like `valid_input(G) :- is_list(G).`

The predicate should capture the distinguishing properties of these inputs, such as:
- Specific dimensions or dimension relationships
- Required colors or color patterns
- Structural properties (symmetry, objects, regions)
- Cell-to-cell relationships

## Useful Prolog Patterns

```prolog
% Check grid dimensions
grid_size(Grid, Rows, Cols) :-
    length(Grid, Rows),
    Grid = [FirstRow|_],
    length(FirstRow, Cols).

% Check all rows have same length
rectangular(Grid) :-
    maplist(length, Grid, Lengths),
    msort(Lengths, [_]).

% Access cell at (Row, Col) - 0-indexed
cell(Grid, Row, Col, Value) :-
    nth0(Row, Grid, RowList),
    nth0(Col, RowList, Value).

% Check all cells satisfy predicate
all_cells(Grid, Pred) :-
    maplist(maplist(Pred), Grid).

% Check value is valid color (0-9)
valid_color(V) :- V >= 0, V =< 9.

% Flatten grid to list of values
flatten_grid(Grid, Values) :-
    append(Grid, Values).

% Count occurrences of value in grid
count_value(Grid, Value, Count) :-
    flatten_grid(Grid, Values),
    include(=(Value), Values, Matches),
    length(Matches, Count).
```

## Output Format

Provide your Prolog code in a ```prolog code block. Define `valid_input/1` as the main entry point with helper predicates as needed.

```prolog
% Helper predicates
check_dimensions(Grid, ExpectedRows, ExpectedCols) :-
    length(Grid, ExpectedRows),
    Grid = [FirstRow|_],
    length(FirstRow, ExpectedCols).

% Main predicate
valid_input(Grid) :-
    % Add your constraints here
    check_dimensions(Grid, Rows, Cols),
    % ... more constraints
    true.
```

Begin your analysis now.
