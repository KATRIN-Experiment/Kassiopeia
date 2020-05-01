#include "KFMPinchonGMatrixCalculator.hh"

namespace KEMField
{


KFMPinchonGMatrixCalculator::KFMPinchonGMatrixCalculator()
{
    fDegree = 0;
    fMatrixType = 0;
}

KFMPinchonGMatrixCalculator::~KFMPinchonGMatrixCalculator() {}

bool KFMPinchonGMatrixCalculator::ComputeMatrix(kfm_matrix* G) const
{
    if (G == nullptr) {
        return false;
    };

    switch (fMatrixType) {
        case 0:
            return ComputeGX(G);
            break;
        case 1:
            return ComputeGY(G);
            break;
        case 2:
            return ComputeGZ(G);
            break;
        case 3:
            return ComputeGZHat(G);
            break;
        case 4:
            return ComputeGZHatInverse(G);
            break;
        default:
            return false;
            break;
    }
}

bool KFMPinchonGMatrixCalculator::ComputeGX(kfm_matrix* G) const
{
    if (CheckMatrixDim(G)) {
        kfm_matrix_set_zero(G);

        if (fDegree > 0) {
            double elem;
            auto l = (double) fDegree;
            double kd;
            unsigned int row;
            unsigned int column;

            for (unsigned int k = 1; k <= fDegree - 1; k++) {
                kd = (double) k;
                elem = (std::sqrt(kd * (kd + 1.0))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));

                row = 2 + k;
                column = k;
                kfm_matrix_set(G, row - 1, column - 1, elem);

                row = 2 * fDegree + 2 - k;
                column = 2 * fDegree + 2 - k;
                kfm_matrix_set(G, row - 1, column - 1, elem);
            }

            for (unsigned int k = 1; k <= fDegree; k++) {
                kd = (double) k;
                elem = (std::sqrt((2 * l + 2 - kd) * (2 * l + 3 - kd))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));

                row = k;
                column = k;
                kfm_matrix_set(G, row - 1, column - 1, -1 * elem);


                row = 2 * fDegree + 4 - k;
                column = 2 * fDegree + 2 - k;
                kfm_matrix_set(G, row - 1, column - 1, -1 * elem);
            }

            row = fDegree + 2;
            column = fDegree + 2;
            elem = (std::sqrt(2.0 * l * (l + 1.0))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));
            kfm_matrix_set(G, row - 1, column - 1, elem);

            row = fDegree + 3;
            column = fDegree + 1;
            elem = (std::sqrt(2.0 * (l + 1.0) * (l + 2.0))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));
            kfm_matrix_set(G, row - 1, column - 1, -1 * elem);

            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

bool KFMPinchonGMatrixCalculator::ComputeGY(kfm_matrix* G) const
{

    if (CheckMatrixDim(G)) {
        kfm_matrix_set_zero(G);

        if (fDegree > 0) {
            double elem;
            auto l = (double) fDegree;
            double kd;
            unsigned int row;
            unsigned int column;

            for (unsigned int k = 1; k <= fDegree - 1; k++) {
                kd = (double) k;
                elem = (std::sqrt(kd * (kd + 1.0))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));

                row = 2 * fDegree + 2 - k;
                column = k;

                kfm_matrix_set(G, row - 1, column - 1, elem);

                row = 2 + k;
                column = 2 * fDegree + 2 - k;
                kfm_matrix_set(G, row - 1, column - 1, -1 * elem);
            }

            for (unsigned int k = 1; k <= fDegree; k++) {
                kd = (double) k;
                elem = (std::sqrt((2 * l + 2 - kd) * (2 * l + 3 - kd))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));

                row = k;
                column = 2 * fDegree + 2 - k;
                kfm_matrix_set(G, row - 1, column - 1, -1 * elem);


                row = 2 * fDegree + 4 - k;
                column = k;
                kfm_matrix_set(G, row - 1, column - 1, elem);
            }

            row = fDegree + 2;
            column = fDegree;
            elem = (std::sqrt(2.0 * l * (l + 1.0))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));
            kfm_matrix_set(G, row - 1, column - 1, elem);

            row = fDegree + 1;
            column = fDegree + 1;
            elem = (std::sqrt(2.0 * (l + 1.0) * (l + 2.0))) / (2.0 * std::sqrt((2 * l + 1) * (2 * l + 3)));
            kfm_matrix_set(G, row - 1, column - 1, -1 * elem);

            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

bool KFMPinchonGMatrixCalculator::ComputeGZ(kfm_matrix* G) const
{
    if (CheckMatrixDim(G)) {
        kfm_matrix_set_zero(G);

        if (fDegree > 0) {
            double elem;
            auto l = (double) fDegree;
            unsigned int row;
            unsigned int column;

            for (unsigned int k = 1; k <= 2 * fDegree + 1; k++) {
                auto kd = (double) k;
                elem = (std::sqrt(kd * (2 * l + 2 - kd))) / (std::sqrt((2 * l + 1) * (2 * l + 3)));
                row = k + 1;
                column = k;
                //subtract off 1 from row and column indices b/c matrix is zero based
                kfm_matrix_set(G, row - 1, column - 1, elem);
            }

            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

bool KFMPinchonGMatrixCalculator::ComputeGZHat(kfm_matrix* G) const
{

    if (CheckHatMatrixDim(G)) {
        kfm_matrix_set_zero(G);

        if (fDegree > 0) {
            double elem;
            auto l = (double) fDegree;
            unsigned int row;
            unsigned int column;

            for (unsigned int k = 1; k <= 2 * fDegree + 1; k++) {
                auto kd = (double) k;
                elem = (std::sqrt(kd * (2 * l + 2 - kd))) / (std::sqrt((2 * l + 1) * (2 * l + 3)));
                row = k;
                column = k;
                //subtract off 1 from row and column indices b/c matrix is zero based
                kfm_matrix_set(G, row - 1, column - 1, elem);
            }

            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

bool KFMPinchonGMatrixCalculator::ComputeGZHatInverse(kfm_matrix* G) const
{
    if (CheckHatMatrixDim(G)) {
        kfm_matrix_set_zero(G);

        if (fDegree > 0) {
            double elem;
            auto l = (double) fDegree;
            unsigned int row;
            unsigned int column;

            for (unsigned int k = 1; k <= 2 * fDegree + 1; k++) {
                auto kd = (double) k;
                elem = (std::sqrt(kd * (2 * l + 2 - kd))) / (std::sqrt((2 * l + 1) * (2 * l + 3)));
                row = k;
                column = k;
                //subtract off 1 from row and column indices b/c matrix is zero based
                kfm_matrix_set(G, row - 1, column - 1, 1.0 / elem);
            }

            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
}

bool KFMPinchonGMatrixCalculator::CheckMatrixDim(const kfm_matrix* G) const
{
    unsigned int nrows = (2 * fDegree + 3);
    unsigned int ncolumns = (2 * fDegree + 1);

    if (nrows != G->size1) {
        return false;
    };
    if (ncolumns != G->size2) {
        return false;
    };

    return true;
}

bool KFMPinchonGMatrixCalculator::CheckHatMatrixDim(const kfm_matrix* G) const
{
    unsigned int nrows = (2 * fDegree + 1);
    unsigned int ncolumns = (2 * fDegree + 1);

    if (nrows != G->size1) {
        return false;
    };
    if (ncolumns != G->size2) {
        return false;
    };

    return true;
}


}  // namespace KEMField
