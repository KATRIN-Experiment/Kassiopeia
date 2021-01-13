#include "KFMScalarMultipoleExpansion.hh"

#include "KFMMessaging.hh"

#include <sstream>

using namespace KEMField;

namespace KEMField
{


const int KFMScalarMultipoleExpansion::fDegreeTable[MULTIPOLE_INDEX_TABLE_SIZE] = {
    0,                                                                                                           //L=0
    1,  1,  1,                                                                                                   //L=1
    2,  2,  2,  2,  2,                                                                                           //L=2
    3,  3,  3,  3,  3,  3,  3,                                                                                   //L=3
    4,  4,  4,  4,  4,  4,  4,  4,  4,                                                                           //L=4
    5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,                                                                   //L=5
    6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,                                                           //L=6
    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,                                                   //L=7
    8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,                                           //L=8
    9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,                                   //L=9
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,                          //L=10
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,                  //L=11
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,          //L=12
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,  //L=13
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14,  //L=14
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15,  //L=15
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
    16, 16, 16, 16, 16, 16,  //L=16
    17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    17, 17, 17, 17, 17, 17, 17, 17,  //L=17
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
    18, 18, 18, 18, 18, 18, 18, 18, 18, 18,  //L=18
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,  //L=19
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20  //L=20
};

const int KFMScalarMultipoleExpansion::fOrderTable[MULTIPOLE_INDEX_TABLE_SIZE] =
    {
        0,                                                                                                      //L=0
        -1,  0,   1,                                                                                            //L=1
        -2,  -1,  0,   1,   2,                                                                                  //L=2
        -3,  -2,  -1,  0,   1,   2,   3,                                                                        //L=3
        -4,  -3,  -2,  -1,  0,   1,   2,   3,   4,                                                              //L=4
        -5,  -4,  -3,  -2,  -1,  0,   1,   2,   3,   4,   5,                                                    //L=5
        -6,  -5,  -4,  -3,  -2,  -1,  0,   1,   2,   3,   4,   5,  6,                                           //L=6
        -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,   1,   2,   3,   4,  5,  6,  7,                                   //L=7
        -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,   1,   2,   3,  4,  5,  6,  7,  8,                           //L=8
        -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,   1,   2,  3,  4,  5,  6,  7,  8,  9,                   //L=9
        -10, -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,   1,  2,  3,  4,  5,  6,  7,  8,  9,  10,          //L=10
        -11, -10, -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,  //L=11
        -12, -11, -10, -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
        11,  12,  //L=12
        -13, -12, -11, -10, -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
        10,  11,  12,  13,  //L=13
        -14, -13, -12, -11, -10, -9,  -8,  -7,  -6,  -5,  -4,  -3, -2, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,
        9,   10,  11,  12,  13,  14,  //L=14
        -15, -14, -13, -12, -11, -10, -9,  -8,  -7,  -6,  -5,  -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  6,  7,
        8,   9,   10,  11,  12,  13,  14,  15,  //L=15
        -16, -15, -14, -13, -12, -11, -10, -9,  -8,  -7,  -6,  -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  6,
        7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  //L=16
        -17, -16, -15, -14, -13, -12, -11, -10, -9,  -8,  -7,  -6, -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,
        6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  //L=17
        -18, -17, -16, -15, -14, -13, -12, -11, -10, -9,  -8,  -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,
        5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16, 17, 18,  //L=18
        -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9,  -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,  3,
        4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15, 16, 17, 18, 19,  //L=19
        -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,
        3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14, 15, 16, 17, 18, 19, 20  //L=20
};


KFMScalarMultipoleExpansion::KFMScalarMultipoleExpansion()
{
    fMomentsReal.resize(0);
    fMomentsImag.resize(0);
}

KFMScalarMultipoleExpansion::~KFMScalarMultipoleExpansion()
{
    ;
}

void KFMScalarMultipoleExpansion::SetNumberOfTermsInSeries(unsigned int n_terms)
{
    if (n_terms > 0) {
        //compute the nearest integer square root
        auto degree_plus_one = (int) std::floor(std::sqrt((double) n_terms));
        unsigned int expected_size = (degree_plus_one) * (degree_plus_one);

        if (expected_size != n_terms) {
            //error, abort!
        }

        int size = TriangleNumber(degree_plus_one);

        fMomentsReal.resize(size);
        fMomentsImag.resize(size);

        Clear();
    }
}

unsigned int KFMScalarMultipoleExpansion::GetNumberOfTermsInSeries() const
{
    int degree = GetDegree();
    return (degree + 1) * (degree + 1);
}


int KFMScalarMultipoleExpansion::GetDegree() const
{
    int storage_size = fMomentsReal.size();
    //now invert the triangle number
    int degree = (int) ((std::sqrt(1 + 8 * storage_size) - 1) / 2) - 1;
    return degree;
}

unsigned int KFMScalarMultipoleExpansion::GetNumberOfTermsFromDegree(unsigned int degree)
{
    return (degree + 1) * (degree + 1);
}

unsigned int KFMScalarMultipoleExpansion::GetDegreeFromNumberOfTerms(unsigned int n_terms)
{
    int degree = (int) (std::floor(std::sqrt((double) n_terms + 0.001))) - 1;
    if (degree > 0) {
        return (unsigned int) degree;
    }
    else {
        return 0;
    }
}


void KFMScalarMultipoleExpansion::SetDegree(const int& l_max)
{
    int degree = std::abs(l_max);
    int size = TriangleNumber(degree + 1);

    fMomentsReal.resize(size);
    fMomentsImag.resize(size);

    Clear();
}

void KFMScalarMultipoleExpansion::Clear()
{
    for (double& i : fMomentsReal) {
        i = 0;
    }

    for (double& i : fMomentsImag) {
        i = 0;
    }
}

void KFMScalarMultipoleExpansion::SetMoments(const std::vector<std::complex<double>>* mom)
{
    int degree = GetDegree();
    unsigned int total_size = (degree + 1) * (degree + 1);

    if (mom->size() == total_size) {
        int psi;

        for (int l = 0; l <= degree; l++) {
            for (int m = 0; m <= l; m++) {
                psi = ComplexBasisIndex(l, m);
                fMomentsReal[RealBasisIndex(l, m)] = ((*mom)[psi]).real();
                fMomentsImag[RealBasisIndex(l, m)] = ((*mom)[psi]).imag();
            }
        }
    }
    else {
        std::stringstream ss;
        ss << "Moments vector not of the expected size.";
        ss << " Expected a vector of size: " << total_size;
        ss << " Recieved a vector of size: " << mom->size();
        kfmout << "KFMScalarMultipoleExpansion::SetMoments: " << ss.str() << std::endl;
    }
}

void KFMScalarMultipoleExpansion::GetMoments(std::vector<std::complex<double>>* mom) const
{
    int degree = GetDegree();
    unsigned int total_size = (degree + 1) * (degree + 1);
    mom->resize(total_size);

    int psi;
    int nsi;
    double real;
    double imag;

    for (int l = 0; l <= degree; l++) {
        for (int m = 0; m <= l; m++) {
            psi = ComplexBasisIndex(l, m);
            nsi = ComplexBasisIndex(l, -m);
            real = fMomentsReal[RealBasisIndex(l, m)];
            imag = fMomentsImag[RealBasisIndex(l, m)];
            (*mom)[psi] = std::complex<double>(real, imag);
            (*mom)[nsi] = std::complex<double>(real, -1 * imag);
        }
    }
}

int KFMScalarMultipoleExpansion::ComplexDegreeReverseLookUp(int storage_index)
{
    if (storage_index < MULTIPOLE_INDEX_TABLE_SIZE) {
        return fDegreeTable[storage_index];
    }
    else {
        return (int) std::floor(std::sqrt((double) storage_index));
    }
}

int KFMScalarMultipoleExpansion::ComplexOrderReverseLookUp(int storage_index)
{
    if (storage_index < MULTIPOLE_INDEX_TABLE_SIZE) {
        return fOrderTable[storage_index];
    }
    else {
        int degree = ComplexDegreeReverseLookUp(storage_index);
        return storage_index - degree * (degree + 1);
    }
}

void KFMScalarMultipoleExpansion::MultiplyByScalar(double scale)
{
    for (unsigned int i = 0; i < fMomentsReal.size(); i++) {
        fMomentsReal[i] *= scale;
        fMomentsImag[i] *= scale;
    }
}

void KFMScalarMultipoleExpansion::SetRealMoments(const std::vector<double>* real_moments)
{
    fMomentsReal = *real_moments;
}

void KFMScalarMultipoleExpansion::SetImaginaryMoments(const std::vector<double>* imag_moments)
{
    fMomentsImag = *imag_moments;
}

void KFMScalarMultipoleExpansion::DefineOutputNode(KSAOutputNode* node) const
{
    if (node != nullptr) {
        node->AddChild(new KSAAssociatedPassedPointerPODOutputNode<KFMScalarMultipoleExpansion,
                                                                   std::vector<double>,
                                                                   &KFMScalarMultipoleExpansion::GetRealMoments>(
            std::string("real"),
            this));
        node->AddChild(new KSAAssociatedPassedPointerPODOutputNode<KFMScalarMultipoleExpansion,
                                                                   std::vector<double>,
                                                                   &KFMScalarMultipoleExpansion::GetImaginaryMoments>(
            std::string("imag"),
            this));
    }
}

void KFMScalarMultipoleExpansion::DefineInputNode(KSAInputNode* node)
{
    if (node != nullptr) {
        node->AddChild(
            new KSAAssociatedPointerPODInputNode<KFMScalarMultipoleExpansion,
                                                 std::vector<double>,
                                                 &KFMScalarMultipoleExpansion::SetRealMoments>(std::string("real"),
                                                                                               this));
        node->AddChild(
            new KSAAssociatedPointerPODInputNode<KFMScalarMultipoleExpansion,
                                                 std::vector<double>,
                                                 &KFMScalarMultipoleExpansion::SetImaginaryMoments>(std::string("imag"),
                                                                                                    this));
    }
}


}  // namespace KEMField
