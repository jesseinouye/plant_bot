#include <Matrix.h>
#include <stdio.h>
#include <initializer_list>

/************************************
 * Private Functions
*/
// Swaps the members of two matrices 
void Matrix::swap(Matrix &m){
    int old_nCols = nCols;
    int old_nRows = nRows;
    nCols = m.nCols;
    nRows = m.nRows;
    m.nCols = old_nCols;
    m.nRows = old_nRows;

    int** old_matrix = matrix;
    matrix = m.matrix;
    m.matrix = old_matrix;
}

/************************************
 * Public Functions
*/
Matrix::Matrix(int _nRows, int _nCols, std::initializer_list<int> list) {
    nRows = _nRows;
    nCols = _nCols;

    // allocate (nRows) number of pointers for arrays
    matrix = new int* [nRows];
    if (matrix == nullptr)
        fprintf(stderr, "Error: Could not allocate memory!");

    // allocate arrays with (nCols) number of elements
    for (int i = 0; i < nRows; i++)
    {
        int** row = matrix + i;
        *row = new int [nCols];
        if (*row == nullptr)
            fprintf(stderr, "Error: Could not allocate memory!");
    }

    if (list.size() < static_cast<size_t>(nRows * nCols)) {
        printf("Error: Not enough elements specified to initialize %dx%d matrix\n", nRows, nCols);
        return;
    }

    // Set desired value
    for (int i = 0; i < nRows; i++){
        for (int j = 0; j < nCols; j++) {
            matrix[i][j] = *(list.begin() + (i * nCols) + j);
        }
    }
}
Matrix::Matrix(int _nRows, int _nCols, int init_value) {
    nRows = _nRows;
    nCols = _nCols;

    // allocate (nRows) number of pointers for arrays
    matrix = new int* [nRows];
    if (matrix == nullptr)
        fprintf(stderr, "Error: Could not allocate memory!");

    // allocate arrays with (nCols) number of elements
    for (int i = 0; i < nRows; i++)
    {
        int** row = matrix + i;
        *row = new int [nCols];
        if (*row == nullptr)
            fprintf(stderr, "Error: Could not allocate memory!");
    }

    // Set desired value
    for (int i = 0; i < nRows; i++){
        for (int j = 0; j < nCols; j++) {
            matrix[i][j] = init_value;
        }
    }
}
Matrix::Matrix(int _nRows, int _nCols) : Matrix(_nRows, _nCols, 0) {};

// Copy Constructor
Matrix::Matrix(Matrix &m) : Matrix(m.nRows, m.nCols) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrix[i][j] = m.matrix[i][j];
        }
    }
}
Matrix::~Matrix()
{
    // de-allocate arrays (rows)
    for (int i = 0; i < nRows; i++)
    {
        int** row = matrix + i;
        delete[] *row;
    }
    // de-allocate array of pointers
    delete[] matrix;
}

void Matrix::print(void) {
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void Matrix::print_detailed(void) { // Note this function is silly and not optimized
    printf("Matrix: %dx%d\n", nRows, nCols);
    for (int i = 0; i < (3 + (nCols * 4) + 3); i++)
    {
        printf("_");
    }
    printf("\n");
    for (int i = 0; i < nRows; i++) {
        printf("| ");
        for (int j = 0; j < nCols; j++) {
            printf("%-4d ", matrix[i][j]);
        }
        printf(" |\n");
    }
    for (int i = 0; i < (3 + (nCols * 4) + 3); i++)
    {
        printf("‾");
    }
    printf("\n");
}

void Matrix::set(std::initializer_list<int> list) {
    if (list.size() < static_cast<size_t>(nRows * nCols)) {
        printf("Error: Not enough elements specified to initialize %dx%d matrix\n", nRows, nCols);
        return;
    }
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            matrix[i][j] = *(list.begin() + (i * nCols) + j);
        }
    }
}
void Matrix::set_row(int row, int values[]) {
    for (int j = 0; j < nCols; j++) {
        matrix[row][j] = values[j];
    }
}
void Matrix::set_col(int col, int values[]) {
    for (int i = 0; i < nCols; i++) {
        matrix[i][col] = values[i];
    }
}
void Matrix::set_diagonal(int values[]) {
    for (int ij = 0; ij < nCols; ij++) {
        matrix[ij][ij] = values[ij];
    }
}

void Matrix::multiply(Matrix m2) {
    // Operand dimension check
    if (nCols != m2.nRows) {
        fprintf(stderr, "Error: Inner matrix dimensions must match for multiplication");
        return;
    }

    // Create matrix to store results
    Matrix m_result(nRows, m2.nCols);

    for (int i1 = 0; i1 < nRows; i1++) {
        for (int j2 = 0; j2 < m2.nCols; j2++){
            m_result.matrix[i1][j2] = 0;
            for (int inner = 0; inner < nCols; inner++) {
                m_result.matrix[i1][j2] += (matrix[i1][inner] * m2.matrix[inner][j2]);
            }
        }
    }

    // Here we swap the dimensions and pointers w the results matrix 
    // This means when the scope exits, our old matrix data will be cleaned up while our results are saved
    swap(m_result);
}

void Matrix::transpose(void) {
    // We want a square matrix as big as the largest dimension
    int max = nRows >= nCols ? nRows : nCols;
    Matrix m_temp(max, max);

    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            m_temp.matrix[j][i] = matrix[i][j];
        }
    }

    Matrix m_result(nCols, nRows);
    for (int i = 0; i < m_result.nRows; i++) {
        for (int j = 0; j < m_result.nCols; j++) {
            m_result.matrix[i][j] = m_temp.matrix[i][j];
        }
    }

    swap(m_result);
}
