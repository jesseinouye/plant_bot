enum JOINT_TYPES {
    REVOLUTE,
    PRISMATIC
};

class Joint {
public:
    int i;
    JOINT_TYPES type;
    float theta;
    float alpha;
    float d;
    float a;

    Joint(int _i, JOINT_TYPES _type, float _theta, float _alpha, float _d, float _a) : i(_i), type(_type), theta(_theta), alpha(_alpha), d(_d), a(_a) { };
};