#include "KGCoreMessage.hh"
#include "KGInterfaceBuilder.hh"

using namespace KGeoBag;
using namespace katrin;
using namespace std;

int main(int argc, char** argv)
{

    if (argc < 7) {
        cout << "usage: ./EulerAngles <alpha> <beta> <gamma> <x> <y> <z> [<x> <y> <z> ...]" << endl;
        return -1;
    }

    double alpha = stof(argv[1]);
    double beta = stof(argv[2]);
    double gamma = stof(argv[3]);

    KRotation rot;
    rot.SetEulerAnglesInDegrees(alpha, beta, gamma);
    cout << "Matrix: " << rot << endl;

    for (int i = 4; i < argc - 2; i += 3) {
        double x = stof(argv[i]);
        double y = stof(argv[i + 1]);
        double z = stof(argv[i + 2]);

        KThreeVector vec(x, y, z);

        KThreeVector rot_vec = rot * vec;
        cout << "Vector " << vec << " -> " << rot_vec << endl;
    }

    return 0;
}
