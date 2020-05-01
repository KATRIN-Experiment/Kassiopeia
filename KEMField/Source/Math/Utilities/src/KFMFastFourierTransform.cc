#include "KFMFastFourierTransform.hh"

#include "KFMBitReversalPermutation.hh"
#include "KFMFastFourierTransformUtilities.hh"
#include "KFMMessaging.hh"

#include <cmath>
#include <cstring>

namespace KEMField
{


KFMFastFourierTransform::KFMFastFourierTransform()
{
    fIsValid = true;
    fForward = true;
    fInitialized = false;
    fSizeIsPowerOfTwo = false;
    fSizeIsPowerOfThree = false;

    fN = 0;
    fM = 0;
    fPermutation = nullptr;
    fTwiddle = nullptr;
    fConjugateTwiddle = nullptr;
    fScale = nullptr;
    fCirculant = nullptr;
    fWorkspace = nullptr;
}

KFMFastFourierTransform::~KFMFastFourierTransform()
{
    DealocateWorkspace();
}

void KFMFastFourierTransform::SetSize(unsigned int N)
{
    fN = N;
    fSizeIsPowerOfTwo = KFMBitReversalPermutation::IsPowerOfTwo(N);
    fSizeIsPowerOfThree = KFMBitReversalPermutation::IsPowerOfBase(N, 3);
    fM = KFMFastFourierTransformUtilities::ComputeBluesteinArraySize(N);
    fIsValid = false;
}

void KFMFastFourierTransform::SetForward()
{
    fForward = true;
}

void KFMFastFourierTransform::SetBackward()
{
    fForward = false;
}

void KFMFastFourierTransform::Initialize()
{
    if (fInput->GetArraySize() != fN || fOutput->GetArraySize() != fN) {
        fIsValid = false;
    }
    else if (!fInitialized) {
        //initialize
        DealocateWorkspace();
        AllocateWorkspace();

        //compute the permutation arrays and twiddle factors
        if (fSizeIsPowerOfTwo) {
            //use radix-2
            KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(fN, fPermutation);
            KFMFastFourierTransformUtilities::ComputeTwiddleFactors(fN, fTwiddle);
            KFMFastFourierTransformUtilities::ComputeConjugateTwiddleFactors(fN, fConjugateTwiddle);
        }

        if (fSizeIsPowerOfThree) {
            //use radix-3
            KFMBitReversalPermutation::ComputeBitReversedIndices(fN, 3, fPermutation);
            KFMFastFourierTransformUtilities::ComputeTwiddleFactors(fN, fTwiddle);
            KFMFastFourierTransformUtilities::ComputeConjugateTwiddleFactors(fN, fConjugateTwiddle);
        }

        if (!fSizeIsPowerOfThree && !fSizeIsPowerOfTwo) {
            //use Bluestein algorithm
            KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(fM, fPermutation);
            KFMFastFourierTransformUtilities::ComputeTwiddleFactors(fM, fTwiddle);
            KFMFastFourierTransformUtilities::ComputeConjugateTwiddleFactors(fM, fConjugateTwiddle);
            KFMFastFourierTransformUtilities::ComputeBluesteinScaleFactors(fN, fScale);
            KFMFastFourierTransformUtilities::ComputeBluesteinCirculantVector(fN, fM, fTwiddle, fScale, fCirculant);
        }

        fIsValid = true;
        fInitialized = true;
    }
}

///Make a call to execute the FFT plan and perform the transformation
void KFMFastFourierTransform::ExecuteOperation()
{
    if (fIsValid) {
        //if input and output point to the same array, don't bother copying data over
        if (fInput != fOutput) {
            //the arrays are not identical so copy the input over to the output
            std::memcpy((void*) fOutput->GetData(), (void*) fInput->GetData(), fN * sizeof(std::complex<double>));
        }


        if (!fForward)  //for IDFT we conjugate first
        {
            std::complex<double>* data = fOutput->GetData();
            for (unsigned int i = 0; i < fN; i++) {
                data[i] = std::conj(data[i]);
            }
        }

        if (fSizeIsPowerOfTwo) {
            //use radix-2
            KFMBitReversalPermutation::PermuteArray<std::complex<double>>(fN, fPermutation, fOutput->GetData());
            KFMFastFourierTransformUtilities::FFTRadixTwo_DIT(fN, fOutput->GetData(), fTwiddle);
        }

        if (fSizeIsPowerOfThree) {
            //use radix-3
            KFMBitReversalPermutation::PermuteArray<std::complex<double>>(fN, fPermutation, fOutput->GetData());
            KFMFastFourierTransformUtilities::FFTRadixThree(fN, fOutput->GetData(), fTwiddle);
        }

        if (!fSizeIsPowerOfThree && !fSizeIsPowerOfTwo) {
            //use bluestein algorithm for arbitrary N
            KFMFastFourierTransformUtilities::FFTBluestein(fN,
                                                           fM,
                                                           fOutput->GetData(),
                                                           fTwiddle,
                                                           fConjugateTwiddle,
                                                           fScale,
                                                           fCirculant,
                                                           fWorkspace);
        }

        if (!fForward)  //for IDFT we conjugate again
        {
            std::complex<double>* data = fOutput->GetData();
            for (unsigned int i = 0; i < fN; i++) {
                data[i] = std::conj(data[i]);
            }
        }
    }
    else {
        //warning
        kfmout << "KFMFastFourierTransform::ExecuteOperation: Warning, transform not valid. Aborting." << kfmendl;
    }
}

void KFMFastFourierTransform::AllocateWorkspace()
{
    if (!fSizeIsPowerOfTwo && !fSizeIsPowerOfThree) {
        //can't perform an in-place transform, need workspace
        fPermutation = new unsigned int[fM];
        fTwiddle = new std::complex<double>[fM];
        fConjugateTwiddle = new std::complex<double>[fM];
        fScale = new std::complex<double>[fN];
        fCirculant = new std::complex<double>[fM];
        fWorkspace = new std::complex<double>[fM];
    }
    else {
        //can do an in-place transform,
        //only need space for the permutation array, and twiddle factors
        fPermutation = new unsigned int[fN];
        fTwiddle = new std::complex<double>[fN];
        fConjugateTwiddle = new std::complex<double>[fN];
    }
}

void KFMFastFourierTransform::DealocateWorkspace()
{
    delete[] fPermutation;
    fPermutation = nullptr;
    delete[] fTwiddle;
    fTwiddle = nullptr;
    delete[] fConjugateTwiddle;
    fConjugateTwiddle = nullptr;
    delete[] fScale;
    fScale = nullptr;
    delete[] fCirculant;
    fCirculant = nullptr;
    delete[] fWorkspace;
    fWorkspace = nullptr;
}

}  // namespace KEMField
