#include <Joint.h>
#include <Matrix.h>

class Manipulator {
    char name[32] = "\n";
    const int n_DOF;

    Joint* joints;

    Matrix T;
    Matrix J;
    


public:
    Manipulator();
    ~Manipulator();

    
};