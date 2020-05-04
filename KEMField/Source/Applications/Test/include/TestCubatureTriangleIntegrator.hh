// Functions for computing electric potential and field with 7-point and 12-point Gaussian cubature
// These functions only use intrinsic C++ data types and is intended to be used for speed comparisons.
//
// Date: 2015/02/23
// Authors: Ferenc Glueck (Cubature code)
//          Daniel Hilk (Code realization)

//////////////////////
// 7-point cubature //
//////////////////////

void GaussPoints_Tri7P(const double* data, double* Q)
{
    const double fCubc3 = 1. / 3.;

    // alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points
    const double fCub7alpha[3] = {fCubc3, 0.059715871789770, 0.797426985353087};
    const double fCub7beta[3] = {fCubc3, 0.470142064105115, 0.101286507323456};
    const double fCub7gamma[3] = {fCubc3, 0.470142064105115, 0.101286507323456};

    // 7 Gaussian points, component-wise, each is weighted average of the corner points A, B, C with the alpha, beta, gamma weights

    const double A[3] = {data[2], data[3], data[4]};
    const double B[3] = {data[2] + (data[0] * data[5]), data[3] + (data[0] * data[6]), data[4] + (data[0] * data[7])};
    const double C[3] = {data[2] + (data[1] * data[8]), data[3] + (data[1] * data[9]), data[4] + (data[1] * data[10])};

    Q[0] = fCub7alpha[0] * A[0] + fCub7beta[0] * B[0] + fCub7gamma[0] * C[0]; /* alpha A, beta B, gamma C - 0 */
    Q[1] = fCub7alpha[0] * A[1] + fCub7beta[0] * B[1] + fCub7gamma[0] * C[1];
    Q[2] = fCub7alpha[0] * A[2] + fCub7beta[0] * B[2] + fCub7gamma[0] * C[2];
    Q[3] = fCub7alpha[1] * A[0] + fCub7beta[1] * B[0] + fCub7gamma[1] * C[0]; /* alpha A, beta B, gamma C - 1 */
    Q[4] = fCub7alpha[1] * A[1] + fCub7beta[1] * B[1] + fCub7gamma[1] * C[1];
    Q[5] = fCub7alpha[1] * A[2] + fCub7beta[1] * B[2] + fCub7gamma[1] * C[2];
    Q[6] = fCub7beta[1] * A[0] + fCub7alpha[1] * B[0] + fCub7gamma[1] * C[0]; /* beta A, alpha B, gamma C - 1 */
    Q[7] = fCub7beta[1] * A[1] + fCub7alpha[1] * B[1] + fCub7gamma[1] * C[1];
    Q[8] = fCub7beta[1] * A[2] + fCub7alpha[1] * B[2] + fCub7gamma[1] * C[2];
    Q[9] = fCub7gamma[1] * A[0] + fCub7beta[1] * B[0] + fCub7alpha[1] * C[0]; /* gamma A, beta B, alpha C - 1 */
    Q[10] = fCub7gamma[1] * A[1] + fCub7beta[1] * B[1] + fCub7alpha[1] * C[1];
    Q[11] = fCub7gamma[1] * A[2] + fCub7beta[1] * B[2] + fCub7alpha[1] * C[2];
    Q[12] = fCub7alpha[2] * A[0] + fCub7beta[2] * B[0] + fCub7gamma[2] * C[0]; /* alpha, beta, gamma - 2 */
    Q[13] = fCub7alpha[2] * A[1] + fCub7beta[2] * B[1] + fCub7gamma[2] * C[1];
    Q[14] = fCub7alpha[2] * A[2] + fCub7beta[2] * B[2] + fCub7gamma[2] * C[2];
    Q[15] = fCub7beta[2] * A[0] + fCub7alpha[2] * B[0] + fCub7gamma[2] * C[0]; /* beta A, alpha B, gamma C - 2 */
    Q[16] = fCub7beta[2] * A[1] + fCub7alpha[2] * B[1] + fCub7gamma[2] * C[1];
    Q[17] = fCub7beta[2] * A[2] + fCub7alpha[2] * B[2] + fCub7gamma[2] * C[2];
    Q[18] = fCub7gamma[2] * A[0] + fCub7beta[2] * B[0] + fCub7alpha[2] * C[0]; /* gamma A, beta B, alpha C - 2 */
    Q[19] = fCub7gamma[2] * A[1] + fCub7beta[2] * B[1] + fCub7alpha[2] * C[1];
    Q[20] = fCub7gamma[2] * A[2] + fCub7beta[2] * B[2] + fCub7alpha[2] * C[2];

    return;
}

void FieldPotTri7P_Cached(double* data, const double* P, double* cub7Q, double* result)
{
    // Gaussian weights

    const double fCub7term1 = 0.225;                               // = 270./1200.
    const double fCub7term2 = 0.00322748612183951407098272116649;  // = sqrt(15)/1200.
    const double fCub7term3 = 0.12916666666666666666666666666667;  // = 155./1200.
    const double fCub7w3[3] = {fCub7term1, fCub7term3 + fCub7term2, fCub7term3 - fCub7term2};
    const double fCub7w[7] = {fCub7w3[0], fCub7w3[1], fCub7w3[1], fCub7w3[1], fCub7w3[2], fCub7w3[2], fCub7w3[2]};

    double partialValue[4];
    double distVector[3];
    double oneOverAbsoluteValue(0.);
    double tmpValue(0.);
    double absoluteValueSquared(0.);
    double componentSquared(0.);
    unsigned short arrayIndex(0);

    // triangle area as defined in class KTriangle: A = 0.5*fA*fB*fN1.Cross(fN2).Magnitude()
    const double triArea =
        0.5 * data[0] * data[1] *
        sqrt(POW2((data[6] * data[10]) - (data[7] * data[9])) + POW2((data[7] * data[8]) - (data[5] * data[10])) +
             POW2((data[5] * data[9]) - (data[6] * data[8])));

    // loop over 7 Gaussian points
    for (unsigned short gaussianPoint = 0; gaussianPoint < 7; gaussianPoint++) {
        absoluteValueSquared = 0.;

        // loop over vector components

        for (unsigned short componentIndex = 0; componentIndex < 3; componentIndex++) {
            // getting index of Gaussian point array cub7Q[21]
            arrayIndex = ((3 * gaussianPoint) + componentIndex);

            // component of distance vector from Gaussian point to computation point ( P - Q_i )
            distVector[componentIndex] = (P[componentIndex] - cub7Q[arrayIndex]);

            // current vector component
            componentSquared = POW2(distVector[componentIndex]);

            // sum up to variable absoluteValueSquared
            absoluteValueSquared += componentSquared;

            // partialValue = P-Q_i, initialization with vector r
            partialValue[componentIndex] = distVector[componentIndex];
        } /* components */

        // separate divisions from computation, here: tmp2 = 1/|P-Q_i|
        oneOverAbsoluteValue = 1. / sqrt(absoluteValueSquared);
        tmpValue = POW3(oneOverAbsoluteValue);
        tmpValue = tmpValue * fCub7w[gaussianPoint];

        // partialValue = partialValue (= vec r) * w_i * (1/|P-Q_i|^3)
        partialValue[0] *= tmpValue;
        partialValue[1] *= tmpValue;
        partialValue[2] *= tmpValue;
        partialValue[3] = fCub7w[gaussianPoint] * oneOverAbsoluteValue;

        // sum up for final result
        result[0] += partialValue[0];
        result[1] += partialValue[1];
        result[2] += partialValue[2];
        result[3] += partialValue[3];
    } /* 7 Gaussian points */

    for (unsigned short i = 0; i < 4; i++) {
        result[i] = triArea * M_ONEOVER_4PI_EPS0 * result[i];
    }

    return;
}

void FieldPotTri7P_NonCached(double* data, const double* P, double* result)
{
    // Gaussian weights

    const double fCub7term1 = 0.225;                               // = 270./1200.
    const double fCub7term2 = 0.00322748612183951407098272116649;  // = sqrt(15)/1200.
    const double fCub7term3 = 0.12916666666666666666666666666667;  // = 155./1200.
    const double fCub7w3[3] = {fCub7term1, fCub7term3 + fCub7term2, fCub7term3 - fCub7term2};
    const double fCub7w[7] = {fCub7w3[0], fCub7w3[1], fCub7w3[1], fCub7w3[1], fCub7w3[2], fCub7w3[2], fCub7w3[2]};

    // triangle area as defined in class KTriangle: A = 0.5*fA*fB*fN1.Cross(fN2).Magnitude()
    const double triArea =
        0.5 * data[0] * data[1] *
        sqrt(POW2((data[6] * data[10]) - (data[7] * data[9])) + POW2((data[7] * data[8]) - (data[5] * data[10])) +
             POW2((data[5] * data[9]) - (data[6] * data[8])));

    // compute Gaussian points
    double cub7Q[21];
    GaussPoints_Tri7P(data, cub7Q);

    double partialValue[4];
    double distVector[3];
    double oneOverAbsoluteValue(0.);
    double tmpValue(0.);
    double absoluteValueSquared(0.);
    double componentSquared(0.);
    unsigned short arrayIndex(0);

    // loop over 7 Gaussian points
    for (unsigned short gaussianPoint = 0; gaussianPoint < 7; gaussianPoint++) {
        absoluteValueSquared = 0.;

        // loop over vector components
        for (unsigned short componentIndex = 0; componentIndex < 3; componentIndex++) {
            // getting index of Gaussian point array cub7Q[21]
            arrayIndex = ((3 * gaussianPoint) + componentIndex);

            // component of distance vector from Gaussian point to computation point ( P - Q_i )
            distVector[componentIndex] = (P[componentIndex] - cub7Q[arrayIndex]);

            // current vector component
            componentSquared = POW2(distVector[componentIndex]);

            // sum up to variable absoluteValueSquared
            absoluteValueSquared += componentSquared;

            // partialValue = P-Q_i, initialization with vector r
            partialValue[componentIndex] = distVector[componentIndex];
        } /* components */

        // separate divisions from computation, here: tmp2 = 1/|P-Q_i|
        oneOverAbsoluteValue = 1. / sqrt(absoluteValueSquared);
        tmpValue = POW3(oneOverAbsoluteValue);
        tmpValue = tmpValue * fCub7w[gaussianPoint];

        // partialValue = partialValue (= vec r) * w_i * (1/|P-Q_i|^3)
        partialValue[0] *= tmpValue;
        partialValue[1] *= tmpValue;
        partialValue[2] *= tmpValue;
        partialValue[3] = fCub7w[gaussianPoint] * oneOverAbsoluteValue;

        // sum up for final result
        result[0] += partialValue[0];
        result[1] += partialValue[1];
        result[2] += partialValue[2];
        result[3] += partialValue[3];
    } /* 7 Gaussian points */

    for (unsigned short i = 0; i < 4; i++) {
        result[i] = triArea * M_ONEOVER_4PI_EPS0 * result[i];
    }
}

///////////////////////
// 12-point cubature //
///////////////////////

void GaussPoints_Tri12P(const double* data, double* Q)
{
    // Calculates the 12 Gaussian points Q[0],...,Q[11]  for the 12-point cubature of the triangle (degree 7)
    // See: K. Gatermann, Computing 40, 229 (1988)
    //  A=T[0], B=T[1], C=T[2]:  corner points of the triangle
    // Area: area of the triangle calculated by Heron's formula
    // alpha, beta, gamma: barycentric (area) coordinates of the Gaussian points;
    // Gaussian point is weighted average of the corner points A, B, C with the alpha, beta ,gamma weights

    const double fCub12alpha[5] = {0.6238226509439084e-1,
                                   0.5522545665692000e-1,
                                   0.3432430294509488e-1,
                                   0.5158423343536001};
    const double fCub12beta[5] = {0.6751786707392436e-1, 0.3215024938520156, 0.6609491961867980, 0.2777161669764050};
    const double fCub12gamma[5] = {0.8700998678316848, 0.6232720494910644, 0.3047265008681072, 0.2064414986699949};

    unsigned short j;

    const double A[3] = {data[2], data[3], data[4]};
    const double B[3] = {data[2] + (data[0] * data[5]), data[3] + (data[0] * data[6]), data[4] + (data[0] * data[7])};
    const double C[3] = {data[2] + (data[1] * data[8]), data[3] + (data[1] * data[9]), data[4] + (data[1] * data[10])};

    j = 0;
    Q[0] = fCub12alpha[j] * A[0] + fCub12beta[j] * B[0] + fCub12gamma[j] * C[0];
    Q[1] = fCub12alpha[j] * A[1] + fCub12beta[j] * B[1] + fCub12gamma[j] * C[1];
    Q[2] = fCub12alpha[j] * A[2] + fCub12beta[j] * B[2] + fCub12gamma[j] * C[2];

    Q[3] = fCub12beta[j] * A[0] + fCub12gamma[j] * B[0] + fCub12alpha[j] * C[0];
    Q[4] = fCub12beta[j] * A[1] + fCub12gamma[j] * B[1] + fCub12alpha[j] * C[1];
    Q[5] = fCub12beta[j] * A[2] + fCub12gamma[j] * B[2] + fCub12alpha[j] * C[2];

    Q[6] = fCub12gamma[j] * A[0] + fCub12alpha[j] * B[0] + fCub12beta[j] * C[0];
    Q[7] = fCub12gamma[j] * A[1] + fCub12alpha[j] * B[1] + fCub12beta[j] * C[1];
    Q[8] = fCub12gamma[j] * A[2] + fCub12alpha[j] * B[2] + fCub12beta[j] * C[2];

    j = 1;
    Q[9] = fCub12alpha[j] * A[0] + fCub12beta[j] * B[0] + fCub12gamma[j] * C[0];
    Q[10] = fCub12alpha[j] * A[1] + fCub12beta[j] * B[1] + fCub12gamma[j] * C[1];
    Q[11] = fCub12alpha[j] * A[2] + fCub12beta[j] * B[2] + fCub12gamma[j] * C[2];

    Q[12] = fCub12beta[j] * A[0] + fCub12gamma[j] * B[0] + fCub12alpha[j] * C[0];
    Q[13] = fCub12beta[j] * A[1] + fCub12gamma[j] * B[1] + fCub12alpha[j] * C[1];
    Q[14] = fCub12beta[j] * A[2] + fCub12gamma[j] * B[2] + fCub12alpha[j] * C[2];

    Q[15] = fCub12gamma[j] * A[0] + fCub12alpha[j] * B[0] + fCub12beta[j] * C[0];
    Q[16] = fCub12gamma[j] * A[1] + fCub12alpha[j] * B[1] + fCub12beta[j] * C[1];
    Q[17] = fCub12gamma[j] * A[2] + fCub12alpha[j] * B[2] + fCub12beta[j] * C[2];

    j = 2;
    Q[18] = fCub12alpha[j] * A[0] + fCub12beta[j] * B[0] + fCub12gamma[j] * C[0];
    Q[19] = fCub12alpha[j] * A[1] + fCub12beta[j] * B[1] + fCub12gamma[j] * C[1];
    Q[20] = fCub12alpha[j] * A[2] + fCub12beta[j] * B[2] + fCub12gamma[j] * C[2];

    Q[21] = fCub12beta[j] * A[0] + fCub12gamma[j] * B[0] + fCub12alpha[j] * C[0];
    Q[22] = fCub12beta[j] * A[1] + fCub12gamma[j] * B[1] + fCub12alpha[j] * C[1];
    Q[23] = fCub12beta[j] * A[2] + fCub12gamma[j] * B[2] + fCub12alpha[j] * C[2];

    Q[24] = fCub12gamma[j] * A[0] + fCub12alpha[j] * B[0] + fCub12beta[j] * C[0];
    Q[25] = fCub12gamma[j] * A[1] + fCub12alpha[j] * B[1] + fCub12beta[j] * C[1];
    Q[26] = fCub12gamma[j] * A[2] + fCub12alpha[j] * B[2] + fCub12beta[j] * C[2];

    j = 3;
    Q[27] = fCub12alpha[j] * A[0] + fCub12beta[j] * B[0] + fCub12gamma[j] * C[0];
    Q[28] = fCub12alpha[j] * A[1] + fCub12beta[j] * B[1] + fCub12gamma[j] * C[1];
    Q[29] = fCub12alpha[j] * A[2] + fCub12beta[j] * B[2] + fCub12gamma[j] * C[2];

    Q[30] = fCub12beta[j] * A[0] + fCub12gamma[j] * B[0] + fCub12alpha[j] * C[0];
    Q[31] = fCub12beta[j] * A[1] + fCub12gamma[j] * B[1] + fCub12alpha[j] * C[1];
    Q[32] = fCub12beta[j] * A[2] + fCub12gamma[j] * B[2] + fCub12alpha[j] * C[2];

    Q[33] = fCub12gamma[j] * A[0] + fCub12alpha[j] * B[0] + fCub12beta[j] * C[0];
    Q[34] = fCub12gamma[j] * A[1] + fCub12alpha[j] * B[1] + fCub12beta[j] * C[1];
    Q[35] = fCub12gamma[j] * A[2] + fCub12alpha[j] * B[2] + fCub12beta[j] * C[2];

    return;
}

void FieldPotTri12P_Cached(double* data, const double* P, double* cub12Q, double* result)
{
    const double fCub12w4[4] = {0.2651702815743450e-1,
                                0.4388140871444811e-1,
                                0.2877504278497528e-1,
                                0.6749318700980879e-1};
    const double fCub12w[12] = {fCub12w4[0] * 2.,
                                fCub12w4[0] * 2.,
                                fCub12w4[0] * 2.,
                                fCub12w4[1] * 2.,
                                fCub12w4[1] * 2.,
                                fCub12w4[1] * 2.,
                                fCub12w4[2] * 2.,
                                fCub12w4[2] * 2.,
                                fCub12w4[2] * 2.,
                                fCub12w4[3] * 2.,
                                fCub12w4[3] * 2.,
                                fCub12w4[3] * 2.};

    double partialValue[4];
    double distVector[3];
    double oneOverAbsoluteValue(0.);
    double tmpValue(0.);
    double absoluteValueSquared(0.);
    double componentSquared(0.);
    unsigned short arrayIndex(0);

    // triangle area as defined in class KTriangle: A = 0.5*fA*fB*fN1.Cross(fN2).Magnitude()

    const double triArea =
        0.5 * data[0] * data[1] *
        sqrt(POW2((data[6] * data[10]) - (data[7] * data[9])) + POW2((data[7] * data[8]) - (data[5] * data[10])) +
             POW2((data[5] * data[9]) - (data[6] * data[8])));

    // loop over 12 Gaussian points

    for (unsigned short gaussianPoint = 0; gaussianPoint < 12; gaussianPoint++) {
        absoluteValueSquared = 0.;

        // loop over vector components

        for (unsigned short componentIndex = 0; componentIndex < 3; componentIndex++) {
            // getting index of Gaussian point array cub12Q[36]

            arrayIndex = ((3 * gaussianPoint) + componentIndex);

            // component of distance vector from Gaussian point to computation point ( P - Q_i )

            distVector[componentIndex] = (P[componentIndex] - cub12Q[arrayIndex]);

            // current vector component

            componentSquared = POW2(distVector[componentIndex]);

            // sum up to variable absoluteValueSquared

            absoluteValueSquared += componentSquared;

            // partialValue = P-Q_i, initialization with vector r

            partialValue[componentIndex] = distVector[componentIndex];
        } /* components */

        // separate divisions from computation, here: tmp2 = 1/|P-Q_i|

        oneOverAbsoluteValue = 1. / sqrt(absoluteValueSquared);
        tmpValue = POW3(oneOverAbsoluteValue);
        tmpValue = tmpValue * fCub12w[gaussianPoint];

        // partialValue = partialValue (= vec r) * w_i * (1/|P-Q_i|^3)

        partialValue[0] *= tmpValue;
        partialValue[1] *= tmpValue;
        partialValue[2] *= tmpValue;
        partialValue[3] = fCub12w[gaussianPoint] * oneOverAbsoluteValue;

        // sum up for final result

        result[0] += partialValue[0];
        result[1] += partialValue[1];
        result[2] += partialValue[2];
        result[3] += partialValue[3];
    } /* 12 Gaussian points */

    for (unsigned short i = 0; i < 4; i++) {
        result[i] = triArea * M_ONEOVER_4PI_EPS0 * result[i];
    }

    return;
}

void FieldPot12P_NonCached(double* data, const double* P, double* result)
{
    const double fCub12w4[4] = {0.2651702815743450e-1,
                                0.4388140871444811e-1,
                                0.2877504278497528e-1,
                                0.6749318700980879e-1};
    const double fCub12w[12] = {fCub12w4[0] * 2.,
                                fCub12w4[0] * 2.,
                                fCub12w4[0] * 2.,
                                fCub12w4[1] * 2.,
                                fCub12w4[1] * 2.,
                                fCub12w4[1] * 2.,
                                fCub12w4[2] * 2.,
                                fCub12w4[2] * 2.,
                                fCub12w4[2] * 2.,
                                fCub12w4[3] * 2.,
                                fCub12w4[3] * 2.,
                                fCub12w4[3] * 2.};

    // triangle area as defined in class KTriangle: A = 0.5*fA*fB*fN1.Cross(fN2).Magnitude()
    const double triArea =
        0.5 * data[0] * data[1] *
        sqrt(POW2((data[6] * data[10]) - (data[7] * data[9])) + POW2((data[7] * data[8]) - (data[5] * data[10])) +
             POW2((data[5] * data[9]) - (data[6] * data[8])));

    double cub12Q[36];
    GaussPoints_Tri12P(data, cub12Q);

    double partialValue[4];
    double distVector[3];
    double oneOverAbsoluteValue(0.);
    double tmpValue(0.);
    double absoluteValueSquared(0.);
    double componentSquared(0.);
    unsigned short arrayIndex(0);

    // loop over 12 Gaussian points
    for (unsigned short gaussianPoint = 0; gaussianPoint < 12; gaussianPoint++) {
        absoluteValueSquared = 0.;

        // loop over vector components
        for (unsigned short componentIndex = 0; componentIndex < 3; componentIndex++) {
            // getting index of Gaussian point array cub12Q[36]
            arrayIndex = ((3 * gaussianPoint) + componentIndex);

            // component of distance vector from Gaussian point to computation point ( P - Q_i )
            distVector[componentIndex] = (P[componentIndex] - cub12Q[arrayIndex]);

            // current vector component
            componentSquared = POW2(distVector[componentIndex]);

            // sum up to variable absoluteValueSquared
            absoluteValueSquared += componentSquared;

            // partialValue = P-Q_i, initialization with vector r
            partialValue[componentIndex] = distVector[componentIndex];
        } /* components */

        // separate divisions from computation, here: tmp2 = 1/|P-Q_i|
        oneOverAbsoluteValue = 1. / sqrt(absoluteValueSquared);
        tmpValue = POW3(oneOverAbsoluteValue);
        tmpValue = tmpValue * fCub12w[gaussianPoint];

        // partialValue = partialValue (= vec r) * w_i * (1/|P-Q_i|^3)
        partialValue[0] *= tmpValue;
        partialValue[1] *= tmpValue;
        partialValue[2] *= tmpValue;
        partialValue[3] = fCub12w[gaussianPoint] * oneOverAbsoluteValue;

        // sum up for final result
        result[0] += partialValue[0];
        result[1] += partialValue[1];
        result[2] += partialValue[2];
        result[3] += partialValue[3];
    } /* 12 Gaussian points */

    for (unsigned short i = 0; i < 4; i++) {
        result[i] = triArea * M_ONEOVER_4PI_EPS0 * result[i];
    }

    return;
}
