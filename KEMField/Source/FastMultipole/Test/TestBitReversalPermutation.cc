#include "KFMBitReversalPermutation.hh"
#include "KFMMessaging.hh"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int N = 16;
    unsigned int index_arr[N];
    KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(N, index_arr);

    for (unsigned int i = 0; i < N; i++) {
        kfmout << "permutation index @ " << i << " = " << index_arr[i] << kfmendl;
    }


    double arr[N];

    for (unsigned int i = 0; i < N; i++) {
        arr[i] = i;
    }

    for (unsigned int i = 0; i < N; i++) {
        kfmout << "original array @ " << i << " = " << arr[i] << kfmendl;
    }

    KFMBitReversalPermutation::PermuteArray<double>(N, index_arr, arr);

    for (unsigned int i = 0; i < N; i++) {
        kfmout << "array after permutation @ " << i << " = " << arr[i] << kfmendl;
    }

    KFMBitReversalPermutation::PermuteArray<double>(N, index_arr, arr);

    for (unsigned int i = 0; i < N; i++) {
        kfmout << "array after double permutation @ " << i << " = " << arr[i] << kfmendl;
    }

    ////////////////////////////////////////////////////////////////////////////
    //now to base 3

    unsigned int N3 = 3;

    unsigned int index_arr3[N3];
    KFMBitReversalPermutation::ComputeBitReversedIndices(N3, 3, index_arr3);

    for (unsigned int i = 0; i < N3; i++) {
        kfmout << "permutation index @ " << i << " = " << index_arr3[i] << kfmendl;
    }


    double arr3[N3];

    for (unsigned int i = 0; i < N3; i++) {
        arr3[i] = i;
    }

    for (unsigned int i = 0; i < N3; i++) {
        kfmout << "original array @ " << i << " = " << arr3[i] << kfmendl;
    }

    KFMBitReversalPermutation::PermuteArray<double>(N3, index_arr3, arr3);

    for (unsigned int i = 0; i < N3; i++) {
        kfmout << "array after permutation @ " << i << " = " << arr3[i] << kfmendl;
    }

    KFMBitReversalPermutation::PermuteArray<double>(N3, index_arr3, arr3);

    for (unsigned int i = 0; i < N3; i++) {
        kfmout << "array after double permutation @ " << i << " = " << arr3[i] << kfmendl;
    }


    //lets factor the number 900 into factors of 2/3/5
    unsigned int n_factors = 3;
    auto* factors = new unsigned int[3];
    auto* powers = new unsigned int[3];
    factors[0] = 2;
    factors[1] = 3;
    factors[2] = 5;


    bool can_factor =
        KFMBitReversalPermutation::Factor(2 * 2 * 2 * 3 * 3 * 3 * 3 * 5 * 5 * 5, n_factors, factors, powers);

    if (can_factor) {

        for (unsigned int i = 0; i < n_factors; i++) {
            kfmout << "power of " << factors[i] << " = " << powers[i] << kfmendl;
        }
    }
    else {
        kfmout << "not factorable" << kfmendl;
    }


    kfmout << "-------------------------" << kfmendl;
    //lets try to factor the number 81 into factors of 2,5,7 (should return false)

    factors[0] = 2;
    factors[1] = 5;
    factors[2] = 7;
    can_factor = KFMBitReversalPermutation::Factor(2 * 5 * 7 * 7 * 7 * 2, n_factors, factors, powers);

    if (can_factor) {
        for (unsigned int i = 0; i < n_factors; i++) {
            kfmout << "power of " << factors[i] << " = " << powers[i] << kfmendl;
        }
    }
    else {
        kfmout << "not factorable" << kfmendl;
    }


    return 0;
}
