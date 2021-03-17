#include "KThreeVector_KEMField.hh"
#include "TVector3.h"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <vector>

// This program tests speed of different classes for math. vectors,
// saved in std::vector container in comparison to one 1-dim. double array.
//
// * std::vector containing TVector3 (ROOT)
// * std::vector containing KThreeVector (Kassiopeia/KGeoBag)
// * double values in std::vector
// * 1-dim. double array
//
// Author: Daniel Hilk

using namespace KEMField;
using namespace KGeoBag;

// for math. on arrays
#define USEHEAP

clock_t start;

void StartTimer()
{
    start = clock();
}

double Time()
{
    double end = clock();
    return ((double) (end - start)) / CLOCKS_PER_SEC;  // time in seconds
    return 0.;
}

int main()
{
    unsigned long N = 5.e7;

    TVector3 rootVector;
    std::vector<TVector3> coRo;

    KThreeVector kasperVector;
    std::vector<KThreeVector> coKa;


    double dblVector = 0.;
    (void) dblVector;
    std::vector<double> coDbl;

    auto* coAr = new double[3 * N]; /*  save double values on heap */

    std::cout << "(0) Initialization of data ... " << std::endl;

    for (unsigned long i = 0; i < N; i++) {
        rootVector.SetXYZ(i, i * i, i * 2.);
        coRo.push_back(rootVector);

        kasperVector.SetComponents(i, i * i, i * 2.);
        coKa.push_back(kasperVector);


        coDbl.push_back(i);
        coDbl.push_back(i * i);
        coDbl.push_back(i * 2.);

        coAr[i * 3] = i;
        coAr[(i * 3) + 1] = i * i;
        coAr[(i * 3) + 2] = i * 2.;
    }

    std::cout << "std::vector<TVector3> size =       " << coRo.size() << std::endl;
    std::cout << "std::vector<KThreeVector> size =   " << coKa.size() << std::endl;
    std::cout << "std::vector<double> size =         " << coDbl.size() << std::endl;
    std::cout << "Items in double array =            " << 3 * N << std::endl << std::endl;

    std::cout << "Time tests with basic vector operations on vectors and arrays. " << std::endl << std::endl;

    std::cout << "(1) Computing cross-product" << std::endl;

    TVector3 rootCross;
    KThreeVector kasperCross;
#ifdef USEHEAP
    auto* dblCross = new double[3];
    auto* arrayCross = new double[3];
#else
    double dblCross[3];
    (void) *dblCross;
    double arrayCross[3];
    (void) *arrayCross;
#endif
    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++)
        rootCross = coRo[l].Cross(coRo[l + 1]);
    std::cout << "std::vector<TVector3> t =          " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++)
        kasperCross = coKa[l].Cross(coKa[l + 1]);
    std::cout << "std::vector<KThreeVector> t =      " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++) {
        dblCross[0] = (coDbl[(l * 3) + 1] * coDbl[((l + 1) * 3) + 2]) - (coDbl[(l * 3) + 2] * coDbl[((l + 1) * 3) + 1]);
        dblCross[1] = (coDbl[(l * 3) + 2] * coDbl[((l + 1) * 3) + 0]) - (coDbl[(l * 3) + 0] * coDbl[((l + 1) * 3) + 2]);
        dblCross[2] = (coDbl[(l * 3) + 0] * coDbl[((l + 1) * 3) + 1]) - (coDbl[(l * 3) + 1] * coDbl[((l + 1) * 3) + 0]);
    }
    std::cout << "std::vector<double> t =            " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++) {
        arrayCross[0] = (coAr[(l * 3) + 1] * coAr[((l + 1) * 3) + 2]) - (coAr[(l * 3) + 2] * coAr[((l + 1) * 3) + 1]);
        arrayCross[1] = (coAr[(l * 3) + 2] * coAr[((l + 1) * 3) + 0]) - (coAr[(l * 3) + 0] * coAr[((l + 1) * 3) + 2]);
        arrayCross[2] = (coAr[(l * 3) + 0] * coAr[((l + 1) * 3) + 1]) - (coAr[(l * 3) + 1] * coAr[((l + 1) * 3) + 0]);
    }
    std::cout << "double[3*N] t =                    " << Time() << std::endl << std::endl;

    std::cout << "(2) scalar * vector" << std::endl;

    TVector3 rootSc;
    KThreeVector kasperSc;
#ifdef USEHEAP
    auto* dblSc = new double[3];
    auto* arraySc = new double[3];
#else
    double dblSc[3];
    (void) *dblSc;
    double arraySc[3];
    (void) *arraySc;
#endif
    StartTimer();
    for (unsigned long l = 0; l < N; l++)
        rootSc = l * coRo[l];
    std::cout << "std::vector<TVector3> t =          " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < N; l++)
        kasperSc = l * coKa[l];
    std::cout << "std::vector<KThreeVector> t =      " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < N; l++) {
        dblSc[0] = l * coDbl[(l * 3)];
        dblSc[1] = l * coDbl[(l * 3) + 1];
        dblSc[2] = l * coDbl[(l * 3) + 2];
    }
    std::cout << "std::vector<double> t =            " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++) {
        arraySc[0] = l * coAr[(l * 3)];
        arraySc[1] = l * coAr[(l * 3) + 1];
        arraySc[2] = l * coAr[(l * 3) + 2];
    }
    std::cout << "double[3*N] t =                    " << Time() << std::endl << std::endl;

    std::cout << "(3) Adding two vectors" << std::endl;

    TVector3 rootAdd;
    KThreeVector kasperAdd;
#ifdef USEHEAP
    auto* dblAdd = new double[3];
    auto* arrayAdd = new double[3];
#else
    double dblAdd[3];
    (void) *dblAdd;
    double arrayAdd[3];
    (void) *arrayAdd;
#endif
    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++)
        rootAdd = coRo[l] + coRo[l + 1];
    std::cout << "std::vector<TVector3> t =          " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++)
        kasperAdd = coKa[l] + coKa[l + 1];
    std::cout << "std::vector<KThreeVector> t =      " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++) {
        dblAdd[0] = coDbl[l * 3] + coDbl[((l + 1) * 3) + 0];
        dblAdd[1] = coDbl[l * 3 + 1] + coDbl[((l + 1) * 3) + 1];
        dblAdd[2] = coDbl[l * 3 + 2] + coDbl[((l + 1) * 3) + 2];
    }
    std::cout << "std::vector<double> t =            " << Time() << std::endl;

    StartTimer();
    for (unsigned long l = 0; l < (N - 1); l++) {
        arrayAdd[0] = coAr[l * 3] + coAr[(l + 1) * 3 + 0];
        arrayAdd[1] = coAr[l * 3 + 1] + coAr[(l + 1) * 3 + 1];
        arrayAdd[2] = coAr[l * 3 + 2] + coAr[(l + 1) * 3 + 2];
    }
    std::cout << "double[3*N] t =                    " << Time() << std::endl << std::endl;

    std::cout << "(4) Magnitude of each vector element" << std::endl;
    double magnitude = 0.;
    (void) magnitude;
    StartTimer();
    for (unsigned long l = 0; l < N; l++)
        magnitude = rootVector.Mag();
    std::cout << "std::vector<TVector3> t =          " << Time() << std::endl;


    StartTimer();
    for (unsigned long l = 0; l < N; l++)
        magnitude = kasperVector.Magnitude();
    std::cout << "std::vector<KThreeVector> t =      " << Time() << std::endl;


    StartTimer();
    for (unsigned long l = 0; l < N; l++)
        magnitude = sqrt((coDbl[(3 * l)] * coDbl[(3 * l)]) + (coDbl[(3 * l) + 1] * coDbl[(3 * l) + 1]) +
                         (coDbl[(3 * l) + 2] * coDbl[(3 * l) + 2]));
    std::cout << "std::vector<double> t =            " << Time() << std::endl;


    StartTimer();
    for (unsigned long l = 0; l < N; l++) {
        magnitude = sqrt((coAr[(3 * l)] * coAr[(3 * l)]) + (coAr[(3 * l) + 1] * coAr[(3 * l) + 1]) +
                         (coAr[(3 * l) + 2] * coAr[(3 * l) + 2]));
    }
    std::cout << "double[3*N] t =                    " << Time() << std::endl;

    delete[] coAr;

    return 0;
}
