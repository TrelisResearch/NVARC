You are an expert Prolog programmer specializing in grid transformations for ARC puzzles.

Your task is to write a Prolog predicate `transform(Input, Output)` that transforms input grids to output grids according to the puzzle rules.

## Grid Format

Grids are lists of lists: `[[1,2],[3,4]]` represents a 2x2 grid.
- Row 0: [1, 2]
- Row 1: [3, 4]
- Colors: 0=BLACK, 1=BLUE, 2=RED, 3=GREEN, 4=YELLOW, 5=GRAY, 6=MAGENTA, 7=ORANGE, 8=SKY, 9=BROWN

## Input Recognizer Code

The following Prolog code validates input grids for this puzzle:

```prolog
{RECOGNIZER_CODE}
```

## Train Input/Output Pairs

Your transform predicate must correctly transform ALL of these:

{TRAIN_PAIRS}

## Your Analysis Process

Before writing code, analyze the transformations inside <analysis> tags:

1. **Dimension changes**: How do output dimensions relate to input dimensions?
2. **Color mapping**: Are colors changed? How?
3. **Spatial operations**: Look for:
   - Rotation (90°, 180°, 270°)
   - Reflection (horizontal, vertical, diagonal)
   - Scaling (up or down)
   - Tiling/repetition
   - Cropping or extraction
4. **Object operations**: Are shapes/objects detected and manipulated?
5. **Cell-level rules**: What determines each output cell's value?

## Requirements

Your predicate must:
1. **Produce exact outputs** for ALL train examples
2. **Generalize** to unseen inputs with the same structure
3. **Be deterministic** - produce one unique output for each input

## Useful Prolog Patterns

```prolog
% Get grid dimensions
grid_size(Grid, Rows, Cols) :-
    length(Grid, Rows),
    (Rows > 0 -> Grid = [R|_], length(R, Cols) ; Cols = 0).

% Access cell at (Row, Col)
cell(Grid, Row, Col, Value) :-
    nth0(Row, Grid, RowList),
    nth0(Col, RowList, Value).

% Create grid of size RxC filled with Value
create_grid(Rows, Cols, Value, Grid) :-
    length(Row, Cols),
    maplist(=(Value), Row),
    length(Grid, Rows),
    maplist(=(Row), Grid).

% Set cell at (Row, Col) to Value
set_cell(GridIn, Row, Col, Value, GridOut) :-
    nth0(Row, GridIn, RowList, RestRows),
    nth0(Col, RowList, _, RestCols),
    nth0(Col, NewRowList, Value, RestCols),
    nth0(Row, GridOut, NewRowList, RestRows).

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

% Rotate 90 degrees clockwise
rotate_90(Grid, Rotated) :-
    transpose(Grid, T),
    flip_horizontal(T, Rotated).

% Map over grid (apply function to each cell)
map_grid(Grid, Pred, Result) :-
    maplist(maplist(Pred), Grid, Result).

% Build output grid cell by cell
build_grid(Rows, Cols, CellPred, Grid) :-
    findall(Row,
        (between(0, Rows1, R), Rows1 is Rows - 1,
         findall(V,
             (between(0, Cols1, C), Cols1 is Cols - 1,
              call(CellPred, R, C, V)),
             Row)),
        Grid).
```

## Output Format

Provide your Prolog code in a ```prolog code block. Define `transform/2` as the main entry point.

```prolog
% Helper predicates (if needed)
extract_pattern(Grid, Pattern) :-
    % ...

% Main transform predicate
transform(Input, Output) :-
    % Determine output dimensions
    grid_size(Input, InRows, InCols),
    OutRows is InRows * 2,
    OutCols is InCols,
    % Build output
    % ...
```

Begin your analysis now.
