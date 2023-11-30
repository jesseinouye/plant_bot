#include <vector>
#include <initializer_list>

class Matrix 
{
private:
    int nRows = 0;
    int nCols = 0;

    void swap(Matrix &m);

public:
    int** matrix = nullptr; //public until overload subscript op

    Matrix(int _nRows, int _nCols, std::initializer_list<int> list);
    Matrix(int _nRows, int _nCols, int init_value);
    Matrix(int _nRows, int _nCols);
    Matrix(Matrix &m);
    ~Matrix();

    //int& operator[] (int index);

    void print(void);
    void print_detailed(void);

    void set(std::initializer_list<int> list);
    void set_row(int row, int values[]);
    void set_col(int col, int values[]);
    void set_diagonal(int values[]);

    void multiply(Matrix m2);
    void transpose(void);

};