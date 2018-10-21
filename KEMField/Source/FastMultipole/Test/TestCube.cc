#include <iostream>
#include <cmath>
#include <iomanip>

#include "KFMArrayMath.hh"
#include "KFMCube.hh"

using namespace KEMField;


int main(int /*argc*/, char** /*argv*/)
{
    KFMCube<1> line;
    KFMCube<2> square;
    KFMCube<3> cube;
    KFMCube<4> hypercube;

    KFMPoint<1> line_center; line_center[0] = 0;
    KFMPoint<2> square_center; square_center[0] = 0; square_center[1] = 0;
    KFMPoint<3> cube_center; cube_center[0] = 0; cube_center[1] = 0; cube_center[2] = 0;
    KFMPoint<4> hypercube_center; hypercube_center[0] = 0; hypercube_center[1] = 0; hypercube_center[2] = 0; hypercube_center[3] = 0;

    double length = 2.0;

    line.SetParameters(line_center, length);
    square.SetParameters(square_center, length);
    cube.SetParameters(cube_center, length);
    hypercube.SetParameters(hypercube_center, length);

    for(unsigned int i=0; i<KFMArrayMath::PowerOfTwo<1>::value; i++)
    {
        KFMPoint<1> c = line.GetCorner(i);
        std::cout<<"line corner @ "<<i<<" = ("<<c[0]<<")"<<std::endl;
    }

    for(unsigned int i=0; i<KFMArrayMath::PowerOfTwo<2>::value; i++)
    {
        KFMPoint<2> c = square.GetCorner(i);
        std::cout<<"square corner @ "<<i<<" = ("<<c[0]<<", "<<c[1]<<")"<<std::endl;
    }

    for(unsigned int i=0; i<KFMArrayMath::PowerOfTwo<3>::value; i++)
    {
        KFMPoint<3> c = cube.GetCorner(i);
        std::cout<<"cube corner @ "<<i<<" = ("<<c[0]<<", "<<c[1]<<", "<<c[2]<<")"<<std::endl;
    }

    for(unsigned int i=0; i<KFMArrayMath::PowerOfTwo<4>::value; i++)
    {
        KFMPoint<4> c = hypercube.GetCorner(i);
        std::cout<<"hypercube corner @ "<<i<<" = ("<<c[0]<<", "<<c[1]<<", "<<c[2]<<", "<<c[3]<<")"<<std::endl;
    }


    return 0;
}
