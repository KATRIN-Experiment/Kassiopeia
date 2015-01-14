#include <iostream>
#include <cmath>
#include <iomanip>

#include <complex>

#include "KFMArrayMath.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{

    unsigned int Dim[5] = {4,5,11,13,12};

    unsigned int test_index[5] = {0,2,7,6,3};
    std::cout<<" the offset  = "<<KFMArrayMath::OffsetFromRowMajorIndex<5>(Dim, test_index)<<std::endl;

    std::cout<<"the size = "<< KFMArrayMath::TotalArraySize<5>(Dim)<<std::endl;

    unsigned int index[5];
    KFMArrayMath::RowMajorIndexFromOffset<5>(KFMArrayMath::OffsetFromRowMajorIndex<5>(Dim, test_index), Dim, index);

    std::cout<<"the test indexes = "<<test_index[0]<<", "<<test_index[1]<<", "<<test_index[2]<<", "<<test_index[3]<<", "<<test_index[4]<<std::endl;
    std::cout<<"the indexes = "<<index[0]<<", "<<index[1]<<", "<<index[2]<<", "<<index[3]<<", "<<index[4]<<std::endl;

    return 0;
}
