You are an expert Prolog programmer.

Your task is to write a Prolog predicate `valid_input(Grid)` that validates whether a given grid is a valid input for this ARC puzzle.

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

## CLP(FD) - Constraint Logic Programming (IMPORTANT)

Use CLP(FD) constraints instead of plain Prolog comparisons for robust validation:

| Instead of... | Use CLP(FD)... | Example |
|---------------|----------------|---------|
| `A == B` | `A #= B` | `A #= B` (A equals B) |
| `A \== B` | `A #\= B` | `A #\= 0` (A not zero) |
| `A >= B` | `A #>= B` | `A #>= 1` (A at least 1) |
| `A =< B` | `A #=< B` | `A #=< 9` (A at most 9) |
| `A > B` | `A #> B` | `A #> 0` (A positive) |
| `A < B` | `A #< B` | `A #< 5` (A less than 5) |

### Loading CLP(FD)
```prolog
:- use_module(library(clpfd)).
```

### Domain Declaration
```prolog
% Declare variables are integers 0-9
[A,B,C,D] ins 0..9.

% Or for a flattened grid
flatten(Grid, Cells),
Cells ins 0..9.
```

### Common CLP(FD) Patterns
```prolog
% Count occurrences of a value in a list
count_value(List, Value, Count) :-
    include(#=(Value), List, Matches),
    length(Matches, Count).

% All elements distinct
all_distinct([A,B,C]).

% Sum constraint
sum(List, #=, Total).
```

## Useful Prolog Patterns

```prolog
% Get grid dimensions
grid_size(Grid, Rows, Cols) :-
    length(Grid, Rows),
    (Rows > 0 -> Grid = [R|_], length(R, Cols) ; Cols = 0).

% Check all rows have same length
uniform_rows([]).
uniform_rows([_]).
uniform_rows([R1,R2|Rest]) :-
    length(R1, L),
    length(R2, L),
    uniform_rows([R2|Rest]).

% Access cell at (Row, Col) - 0-indexed
cell(Grid, Row, Col, Value) :-
    nth0(Row, Grid, RowList),
    nth0(Col, RowList, Value).

% Check value is valid color (0-9)
valid_color(V) :- integer(V), V >= 0, V =< 9.

% Check all cells are valid colors
all_valid_colors(Grid) :-
    flatten(Grid, Cells),
    maplist(valid_color, Cells).

% Check two cells are equal
cells_equal(Grid, R1, C1, R2, C2) :-
    cell(Grid, R1, C1, V),
    cell(Grid, R2, C2, V).

% Check cell has specific value
cell_is(Grid, Row, Col, Value) :-
    cell(Grid, Row, Col, Value).

% Count occurrences of a value
count_value(Grid, Value, Count) :-
    flatten(Grid, Cells),
    include(=(Value), Cells, Matches),
    length(Matches, Count).

% All elements in list are equal
all_equal([]).
all_equal([_]).
all_equal([X,X|Rest]) :- all_equal([X|Rest]).

% All elements are different
all_different([]).
all_different([H|T]) :- \+ member(H, T), all_different(T).

% Get a row
get_row(Grid, RowIdx, Row) :- nth0(RowIdx, Grid, Row).

% Get a column
get_col(Grid, ColIdx, Col) :- maplist(nth0(ColIdx), Grid, Col).

% Check if grid contains a value
contains_value(Grid, Value) :-
    flatten(Grid, Cells),
    member(Value, Cells).

% Transpose grid (swap rows and columns)
transpose([], []).
transpose([[]|_], []).
transpose(Matrix, [Row|Rows]) :-
    maplist(nth0(0), Matrix, Row),
    maplist(select_tail, Matrix, Tails),
    transpose(Tails, Rows).
select_tail([_|T], T).

% Flip grid horizontally (reverse each row)
flip_horizontal(Grid, Flipped) :-
    maplist(reverse, Grid, Flipped).

% Flip grid vertically (reverse row order)
flip_vertical(Grid, Flipped) :-
    reverse(Grid, Flipped).
```

## Example: 3x3 Grid with Diagonal Pattern

For a 3x3 grid where diagonal cells are equal and non-zero:

```prolog
:- use_module(library(clpfd)).

valid_input(Grid) :-
    % Check structure: 3x3 grid
    Grid = [[A,B,C],[D,E,F],[G,H,I]],

    % All cells are valid colors (0-9)
    flatten(Grid, Cells),
    Cells ins 0..9,

    % Constraints: diagonal cells equal (use #=)
    A #= E, E #= I,

    % Diagonal value is non-zero (use #\=)
    A #\= 0,

    % Off-diagonal cells are zero (use #=)
    B #= 0, C #= 0,
    D #= 0, F #= 0,
    G #= 0, H #= 0.
```

## Output Format

Provide your Prolog code in a ```prolog code block. Include:
1. Helper predicates as needed
2. `valid_input/1` as main entry point
3. Clear comments explaining the constraints

Begin your analysis now.
