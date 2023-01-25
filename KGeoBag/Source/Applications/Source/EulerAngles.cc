#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

void Print(const KThreeVector& aXAxis, const KThreeVector& aYAxis, const KThreeVector& aZAxis)
{
    KRotation tRotation;
    double tAlpha, tBeta, tGamma;

    tRotation.SetRotatedFrame(aXAxis, aYAxis, aZAxis);
    tRotation.GetEulerAnglesInDegrees(tAlpha, tBeta, tGamma);
    printf(" X-Axis: %12.9f %12.9f %12.9f\n", aXAxis[0], aXAxis[1], aXAxis[2]);
    printf(" Y-Axis: %12.9f %12.9f %12.9f\n", aYAxis[0], aYAxis[1], aYAxis[2]);
    printf(" Z-Axis: %12.9f %12.9f %12.9f\n", aZAxis[0], aZAxis[1], aXAxis[2]);
    printf(" Angles: %12.6f %12.6f %12.6f\n", tAlpha, tBeta, tGamma);
}

int main(int argc, char** argv)
{

    if (argc < 4) {
        cout << "usage: ./EulerAngles <alpha> <beta> <gamma> [<alpha> <beta> <gamma> ...]" << endl;
        cout << "  Apply one or more Euler angle transformations and print resulting rotation matrix." << endl;
        return -1;
    }

    KThreeVector tXAxis = KThreeVector::sXUnit;
    KThreeVector tYAxis = KThreeVector::sYUnit;
    KThreeVector tZAxis = KThreeVector::sZUnit;

    Print(tXAxis, tYAxis, tZAxis);

    for (int i = 1; i < argc - 2; i += 3) {  // skip first arg
        double tAlpha = stof(argv[i]);
        double tBeta = stof(argv[i + 1]);
        double tGamma = stof(argv[i + 2]);

        cout << "-- " << (i / 3 + 1) << ". transformation: " << KThreeVector(tAlpha, tBeta, tGamma) << endl;

        KTransformation tTransform;
        tTransform.SetRotationEuler(tAlpha, tBeta, tGamma);
        tTransform.ApplyRotation(tXAxis);
        tTransform.ApplyRotation(tYAxis);
        tTransform.ApplyRotation(tZAxis);

        Print(tXAxis, tYAxis, tZAxis);
    }

    return 0;
}
