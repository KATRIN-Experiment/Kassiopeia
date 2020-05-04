#define M_MINDISTANCE           1.E-15
#define M_MINDISTANCETOSIDELINE 1.E-14
#define M_DISTANCECORRECTIONN3  1.E-7 /* step in N3 direction if field point is on edge */
#define M_LOGARGQUOTIENT        1.E-6 /* limit of quotient dist/sM for Taylor expansion (if field point is on line) */

double SolidAngleTriangle(double* data, const double* P)
{
    double res(0.);

    // corner points P0, P1 and P2

    const double triP0[3] = {data[2], data[3], data[4]};

    const double triP1[3] = {data[2] + (data[0] * data[5]),
                             data[3] + (data[0] * data[6]),
                             data[4] + (data[0] * data[7])};  // = fP0 + fN1*fA

    const double triP2[3] = {data[2] + (data[1] * data[8]),
                             data[3] + (data[1] * data[9]),
                             data[4] + (data[1] * data[10])};  // = fP0 + fN2*fB

    // get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = data[6] * data[10] - data[7] * data[9];
    triN3[1] = data[7] * data[8] - data[5] * data[10];
    triN3[2] = data[5] * data[9] - data[6] * data[8];
    const double triMagN3 = 1. / sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
    triN3[0] = triN3[0] * triMagN3;
    triN3[1] = triN3[1] * triMagN3;
    triN3[2] = triN3[2] * triMagN3;

    // triangle centroid

    const double triCenter[3] = {data[2] + (data[0] * data[5] + data[1] * data[8]) / 3.,
                                 data[3] + (data[0] * data[6] + data[1] * data[9]) / 3.,
                                 data[4] + (data[0] * data[7] + data[1] * data[10]) / 3.};

    // quantity h, magnitude corresponds to distance from field point to triangle plane

    const double h =
        (triN3[0] * (P[0] - triCenter[0])) + (triN3[1] * (P[1] - triCenter[1])) + (triN3[2] * (P[2] - triCenter[2]));

    const double triMagCenterToP =
        sqrt(POW2(P[0] - triCenter[0]) + POW2(P[1] - triCenter[1]) + POW2(P[2] - triCenter[2]));

    if (triMagCenterToP <= M_MINDISTANCE)
        res = 2. * M_PI;
    else {
        // unit vectors of distances of corner points to field point in positive rotation order

        double triDistP0Unit[3];
        const double magP0 = 1. / sqrt(POW2(triP0[0] - P[0]) + POW2(triP0[1] - P[1]) + POW2(triP0[2] - P[2]));
        triDistP0Unit[0] = magP0 * (triP0[0] - P[0]);
        triDistP0Unit[1] = magP0 * (triP0[1] - P[1]);
        triDistP0Unit[2] = magP0 * (triP0[2] - P[2]);

        double triDistP1Unit[3];
        const double magP1 = 1. / sqrt(POW2(triP1[0] - P[0]) + POW2(triP1[1] - P[1]) + POW2(triP1[2] - P[2]));
        triDistP1Unit[0] = magP1 * (triP1[0] - P[0]);
        triDistP1Unit[1] = magP1 * (triP1[1] - P[1]);
        triDistP1Unit[2] = magP1 * (triP1[2] - P[2]);

        double triDistP2Unit[3];
        const double magP2 = 1. / sqrt(POW2(triP2[0] - P[0]) + POW2(triP2[1] - P[1]) + POW2(triP2[2] - P[2]));
        triDistP2Unit[0] = magP2 * (triP2[0] - P[0]);
        triDistP2Unit[1] = magP2 * (triP2[1] - P[1]);
        triDistP2Unit[2] = magP2 * (triP2[2] - P[2]);

        const double x = 1. +
                         ((triDistP0Unit[0] * triDistP1Unit[0]) + (triDistP0Unit[1] * triDistP1Unit[1]) +
                          (triDistP0Unit[2] * triDistP1Unit[2])) +
                         ((triDistP0Unit[0] * triDistP2Unit[0]) + (triDistP0Unit[1] * triDistP2Unit[1]) +
                          (triDistP0Unit[2] * triDistP2Unit[2])) +
                         ((triDistP1Unit[0] * triDistP2Unit[0]) + (triDistP1Unit[1] * triDistP2Unit[1]) +
                          (triDistP1Unit[2] * triDistP2Unit[2]));

        // cross product
        const double a12[3] = {(triDistP1Unit[1] * triDistP2Unit[2]) - (triDistP1Unit[2] * triDistP2Unit[1]),
                               (triDistP1Unit[2] * triDistP2Unit[0]) - (triDistP1Unit[0] * triDistP2Unit[2]),
                               (triDistP1Unit[0] * triDistP2Unit[1]) - (triDistP1Unit[1] * triDistP2Unit[0])};

        const double y =
            fabs(((triDistP0Unit[0] * a12[0]) + (triDistP0Unit[1] * a12[1]) + (triDistP0Unit[2] * a12[2])));

        res = fabs(2. * atan2(y, x));
    }

    if (h < 0.)
        res *= -1.;

    return res;
}

double LogArgTaylor(const double sMin, const double dist)
{
    double quotient = fabs(dist / sMin);
    if (quotient < 1.e-14)
        quotient = 1.e-14;

    // Taylor expansion of log argument to second order
    double res = 0.5 * fabs(sMin) * POW2(quotient);

    return res;
}

void IqLFieldAndPotential(double* data, const double* P, const unsigned short countCross,
                          const unsigned short lineIndex, const double dist, double* result)
{
    // corner points P0, P1 and P2

    const double triP0[3] = {data[2], data[3], data[4]};

    const double triP1[3] = {data[2] + (data[0] * data[5]),
                             data[3] + (data[0] * data[6]),
                             data[4] + (data[0] * data[7])};  // = fP0 + fN1*fA

    const double triP2[3] = {data[2] + (data[1] * data[8]),
                             data[3] + (data[1] * data[9]),
                             data[4] + (data[1] * data[10])};  // = fP0 + fN2*fB

    // get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = data[6] * data[10] - data[7] * data[9];
    triN3[1] = data[7] * data[8] - data[5] * data[10];
    triN3[2] = data[5] * data[9] - data[6] * data[8];
    const double triMagN3 = 1. / sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
    triN3[0] = triN3[0] * triMagN3;
    triN3[1] = triN3[1] * triMagN3;
    triN3[2] = triN3[2] * triMagN3;

    // side line unit vectors

    double triAlongSideP0P1Unit[3] = {data[5], data[6], data[7]};  // = N1
    double triAlongSideP1P2Unit[3];

    const double magP1P2 = 1. / sqrt(POW2(triP2[0] - triP1[0]) + POW2(triP2[1] - triP1[1]) + POW2(triP2[2] - triP1[2]));
    triAlongSideP1P2Unit[0] = magP1P2 * (triP2[0] - triP1[0]);
    triAlongSideP1P2Unit[1] = magP1P2 * (triP2[1] - triP1[1]);
    triAlongSideP1P2Unit[2] = magP1P2 * (triP2[2] - triP1[2]);

    double triAlongSideP2P0Unit[3] = {-data[8], -data[9], -data[10]};  // = -N2

    // length values of side lines, only half value is needed

    const double triAlongSideHalfLengthP0P1 = 0.5 * data[0];
    const double triAlongSideHalfLengthP1P2 =
        0.5 * sqrt(POW2(triP2[0] - triP1[0]) + POW2(triP2[1] - triP1[1]) + POW2(triP2[2] - triP1[2]));
    const double triAlongSideHalfLengthP2P0 = 0.5 * data[1];

    // center point of each side to field point

    const double e0[3] = {triAlongSideHalfLengthP0P1 * triAlongSideP0P1Unit[0] + triP0[0],
                          triAlongSideHalfLengthP0P1 * triAlongSideP0P1Unit[1] + triP0[1],
                          triAlongSideHalfLengthP0P1 * triAlongSideP0P1Unit[2] + triP0[2]};

    const double e1[3] = {triAlongSideHalfLengthP1P2 * triAlongSideP1P2Unit[0] + triP1[0],
                          triAlongSideHalfLengthP1P2 * triAlongSideP1P2Unit[1] + triP1[1],
                          triAlongSideHalfLengthP1P2 * triAlongSideP1P2Unit[2] + triP1[2]};

    const double e2[3] = {triAlongSideHalfLengthP2P0 * triAlongSideP2P0Unit[0] + triP2[0],
                          triAlongSideHalfLengthP2P0 * triAlongSideP2P0Unit[1] + triP2[1],
                          triAlongSideHalfLengthP2P0 * triAlongSideP2P0Unit[2] + triP2[2]};

    // outward pointing vector m, perpendicular to side lines

    const double m0[3] = {(triAlongSideP0P1Unit[1] * triN3[2]) - (triAlongSideP0P1Unit[2] * triN3[1]),
                          (triAlongSideP0P1Unit[2] * triN3[0]) - (triAlongSideP0P1Unit[0] * triN3[2]),
                          (triAlongSideP0P1Unit[0] * triN3[1]) - (triAlongSideP0P1Unit[1] * triN3[0])};

    const double m1[3] = {(triAlongSideP1P2Unit[1] * triN3[2]) - (triAlongSideP1P2Unit[2] * triN3[1]),
                          (triAlongSideP1P2Unit[2] * triN3[0]) - (triAlongSideP1P2Unit[0] * triN3[2]),
                          (triAlongSideP1P2Unit[0] * triN3[1]) - (triAlongSideP1P2Unit[1] * triN3[0])};

    const double m2[3] = {(triAlongSideP2P0Unit[1] * triN3[2]) - (triAlongSideP2P0Unit[2] * triN3[1]),
                          (triAlongSideP2P0Unit[2] * triN3[0]) - (triAlongSideP2P0Unit[0] * triN3[2]),
                          (triAlongSideP2P0Unit[0] * triN3[1]) - (triAlongSideP2P0Unit[1] * triN3[0])};

    // size t

    const double t0 = (m0[0] * (P[0] - e0[0])) + (m0[1] * (P[1] - e0[1])) + (m0[2] * (P[2] - e0[2]));
    const double t1 = (m1[0] * (P[0] - e1[0])) + (m1[1] * (P[1] - e1[1])) + (m1[2] * (P[2] - e1[2]));
    const double t2 = (m2[0] * (P[0] - e2[0])) + (m2[1] * (P[1] - e2[1])) + (m2[2] * (P[2] - e2[2]));

    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const double triDistP0[3] = {triP0[0] - P[0], triP0[1] - P[1], triP0[2] - P[2]};

    const double triDistP1[3] = {triP1[0] - P[0], triP1[1] - P[1], triP1[2] - P[2]};

    const double triDistP2[3] = {triP2[0] - P[0], triP2[1] - P[1], triP2[2] - P[2]};

    const double triMagDistP0 = sqrt(POW2(triDistP0[0]) + POW2(triDistP0[1]) + POW2(triDistP0[2]));
    const double triMagDistP1 = sqrt(POW2(triDistP1[0]) + POW2(triDistP1[1]) + POW2(triDistP1[2]));
    const double triMagDistP2 = sqrt(POW2(triDistP2[0]) + POW2(triDistP2[1]) + POW2(triDistP2[2]));

    // evaluation of line integral

    double logArgNom, logArgDenom;
    double iLPhi = 0.;
    double iLField[3] = {0., 0., 0.};
    double tmpScalar;

    // 0 //

    double rM = triMagDistP0;
    double rP = triMagDistP1;
    double sM = (triDistP0[0] * triAlongSideP0P1Unit[0]) + (triDistP0[1] * triAlongSideP0P1Unit[1]) +
                (triDistP0[2] * triAlongSideP0P1Unit[2]);
    double sP = (triDistP1[0] * triAlongSideP0P1Unit[0]) + (triDistP1[1] * triAlongSideP0P1Unit[1]) +
                (triDistP1[2] * triAlongSideP0P1Unit[2]);

    if ((countCross == 1) && (lineIndex == 0)) {
        if (fabs(dist / sM) < M_LOGARGQUOTIENT) {
            logArgNom = (rP + sP);
            tmpScalar = (log(logArgNom) - log(LogArgTaylor(sM, dist)));
            iLField[0] += (m0[0] * tmpScalar);
            iLField[1] += (m0[1] * tmpScalar);
            iLField[2] += (m0[2] * tmpScalar);
            iLPhi += (t0 * tmpScalar);
        }
    }

    if (lineIndex != 0) {
        if ((rM + sM) > (rP - sP)) {
            logArgNom = (rP + sP);
            logArgDenom = (rM + sM);
        }
        else {
            logArgNom = (rM - sM);
            logArgDenom = (rP - sP);
        }

        tmpScalar = (log(logArgNom) - log(logArgDenom));
        iLField[0] += (m0[0] * tmpScalar);
        iLField[1] += (m0[1] * tmpScalar);
        iLField[2] += (m0[2] * tmpScalar);
        iLPhi += (t0 * tmpScalar);
    }

    // 1 //

    rM = triMagDistP1;
    rP = triMagDistP2;
    sM = (triDistP1[0] * triAlongSideP1P2Unit[0]) + (triDistP1[1] * triAlongSideP1P2Unit[1]) +
         (triDistP1[2] * triAlongSideP1P2Unit[2]);
    sP = (triDistP2[0] * triAlongSideP1P2Unit[0]) + (triDistP2[1] * triAlongSideP1P2Unit[1]) +
         (triDistP2[2] * triAlongSideP1P2Unit[2]);

    if ((countCross == 1) && (lineIndex == 1)) {
        if (fabs(dist / sM) < M_LOGARGQUOTIENT) {
            logArgNom = (rP + sP);
            tmpScalar = (log(logArgNom) - log(LogArgTaylor(sM, dist)));
            iLField[0] += (m1[0] * tmpScalar);
            iLField[1] += (m1[1] * tmpScalar);
            iLField[2] += (m1[2] * tmpScalar);
            iLPhi += (t1 * tmpScalar);
        }
    }

    if (lineIndex != 1) {
        if ((rM + sM) > (rP - sP)) {
            logArgNom = (rP + sP);
            logArgDenom = (rM + sM);
        }
        else {
            logArgNom = (rM - sM);
            logArgDenom = (rP - sP);
        }

        tmpScalar = (log(logArgNom) - log(logArgDenom));
        iLField[0] += (m1[0] * tmpScalar);
        iLField[1] += (m1[1] * tmpScalar);
        iLField[2] += (m1[2] * tmpScalar);
        iLPhi += (t1 * tmpScalar);
    }

    // 2 //

    rM = triMagDistP2;
    rP = triMagDistP0;
    sM = (triDistP2[0] * triAlongSideP2P0Unit[0]) + (triDistP2[1] * triAlongSideP2P0Unit[1]) +
         (triDistP2[2] * triAlongSideP2P0Unit[2]);
    sP = (triDistP0[0] * triAlongSideP2P0Unit[0]) + (triDistP0[1] * triAlongSideP2P0Unit[1]) +
         (triDistP0[2] * triAlongSideP2P0Unit[2]);

    if ((countCross == 1) && (lineIndex == 2)) {
        if (fabs(dist / sM) < M_LOGARGQUOTIENT) {
            logArgNom = (rP + sP);
            tmpScalar = (log(logArgNom) - log(LogArgTaylor(sM, dist)));
            iLField[0] += (m2[0] * tmpScalar);
            iLField[1] += (m2[1] * tmpScalar);
            iLField[2] += (m2[2] * tmpScalar);
            iLPhi += (t2 * tmpScalar);
        }
    }

    if (lineIndex != 2) {
        if ((rM + sM) > (rP - sP)) {
            logArgNom = (rP + sP);
            logArgDenom = (rM + sM);
        }
        else {
            logArgNom = (rM - sM);
            logArgDenom = (rP - sP);
        }

        tmpScalar = (log(logArgNom) - log(logArgDenom));
        iLField[0] += (m2[0] * tmpScalar);
        iLField[1] += (m2[1] * tmpScalar);
        iLField[2] += (m2[2] * tmpScalar);
        iLPhi += (t2 * tmpScalar);
    }

    for (unsigned short i = 0; i < 3; i++)
        result[i] = iLField[i];
    result[4] = iLPhi;
}

void FieldPotTriRWG(double* triData, const double* P, double* result)
{
    // corner points P0, P1 and P2

    const double triP0[3] = {triData[2], triData[3], triData[4]};

    const double triP1[3] = {triData[2] + (triData[0] * triData[5]),
                             triData[3] + (triData[0] * triData[6]),
                             triData[4] + (triData[0] * triData[7])};  // = fP0 + fN1*fA

    const double triP2[3] = {triData[2] + (triData[1] * triData[8]),
                             triData[3] + (triData[1] * triData[9]),
                             triData[4] + (triData[1] * triData[10])};  // = fP0 + fN2*fB

    // get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = triData[6] * triData[10] - triData[7] * triData[9];
    triN3[1] = triData[7] * triData[8] - triData[5] * triData[10];
    triN3[2] = triData[5] * triData[9] - triData[6] * triData[8];
    const double triMagN3 = 1. / sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
    triN3[0] = triN3[0] * triMagN3;
    triN3[1] = triN3[1] * triMagN3;
    triN3[2] = triN3[2] * triMagN3;

    // triangle centroid

    const double triCenter[3] = {triData[2] + (triData[0] * triData[5] + triData[1] * triData[8]) / 3.,
                                 triData[3] + (triData[0] * triData[6] + triData[1] * triData[9]) / 3.,
                                 triData[4] + (triData[0] * triData[7] + triData[1] * triData[10]) / 3.};

    // side line vectors

    const double triAlongSideP0P1[3] = {triData[0] * triData[5],
                                        triData[0] * triData[6],
                                        triData[0] * triData[7]};  // = A * N1

    const double triAlongSideP1P2[3] = {triP2[0] - triP1[0], triP2[1] - triP1[1], triP2[2] - triP1[2]};

    const double triAlongSideP2P0[3] = {(-1) * triData[1] * triData[8],
                                        (-1) * triData[1] * triData[9],
                                        (-1) * triData[1] * triData[10]};  // = -B * N2

    // length values of side lines

    const double triAlongSideLengthP0P1 = triData[0];
    const double triAlongSideLengthP1P2 =
        sqrt(POW2(triP2[0] - triP1[0]) + POW2(triP2[1] - triP1[1]) + POW2(triP2[2] - triP1[2]));
    const double triAlongSideLengthP2P0 = triData[1];

    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const double triDistP0[3] = {triP0[0] - P[0], triP0[1] - P[1], triP0[2] - P[2]};
    const double triDistP1[3] = {triP1[0] - P[0], triP1[1] - P[1], triP1[2] - P[2]};
    const double triDistP2[3] = {triP2[0] - P[0], triP2[1] - P[1], triP2[2] - P[2]};

    // check if field point is at triangle edge or at side line

    double distToLine = 0.;
    double distToLineMin = 0.;

    unsigned short correctionLineIndex = 99;  // index of crossing line
    unsigned short correctionCounter = 0;     // point is on triangle edge (= 2) or point is on side line (= 1)
    double lineLambda = -1.;                  /* parameter for distance vector */

    // auxiliary values for distance check

    double tmpVector[3];
    double tmpScalar;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1. / triAlongSideLengthP0P1;

    // compute cross product
    tmpVector[0] = (triAlongSideP0P1[1] * triDistP0[2]) - (triAlongSideP0P1[2] * triDistP0[1]);
    tmpVector[1] = (triAlongSideP0P1[2] * triDistP0[0]) - (triAlongSideP0P1[0] * triDistP0[2]);
    tmpVector[2] = (triAlongSideP0P1[0] * triDistP0[1]) - (triAlongSideP0P1[1] * triDistP0[0]);

    distToLine = sqrt(POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2])) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-triDistP0[0] * triAlongSideP0P1[0]) + (-triDistP0[1] * triAlongSideP0P1[1]) +
                  (-triDistP0[2] * triAlongSideP0P1[2])) *
                 POW2(tmpScalar);

    if (distToLine < M_MINDISTANCETOSIDELINE) {
        if (lineLambda >= 0. && lineLambda <= 1.) {
            distToLineMin = distToLine;
            correctionCounter++;
            correctionLineIndex = 0;
        } /* lambda */
    }     /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1. / triAlongSideLengthP1P2;

    // compute cross product

    tmpVector[0] = (triAlongSideP1P2[1] * triDistP1[2]) - (triAlongSideP1P2[2] * triDistP1[1]);
    tmpVector[1] = (triAlongSideP1P2[2] * triDistP1[0]) - (triAlongSideP1P2[0] * triDistP1[2]);
    tmpVector[2] = (triAlongSideP1P2[0] * triDistP1[1]) - (triAlongSideP1P2[1] * triDistP1[0]);

    distToLine = sqrt(POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2])) * tmpScalar;

    // factor -1 for direction triDistP1 vector

    lineLambda = ((-triDistP1[0] * triAlongSideP1P2[0]) + (-triDistP1[1] * triAlongSideP1P2[1]) +
                  (-triDistP1[2] * triAlongSideP1P2[2])) *
                 POW2(tmpScalar);

    if (distToLine < M_MINDISTANCETOSIDELINE) {
        if (lineLambda >= 0. && lineLambda <= 1.) {
            distToLineMin = distToLine;
            correctionCounter++;
            correctionLineIndex = 1;
        } /* lambda */
    }     /* distance */

    // 2 - check distances to P2P0 side line

    tmpScalar = 1. / triAlongSideLengthP2P0;

    // compute cross product

    tmpVector[0] = (triAlongSideP2P0[1] * triDistP2[2]) - (triAlongSideP2P0[2] * triDistP2[1]);
    tmpVector[1] = (triAlongSideP2P0[2] * triDistP2[0]) - (triAlongSideP2P0[0] * triDistP2[2]);
    tmpVector[2] = (triAlongSideP2P0[0] * triDistP2[1]) - (triAlongSideP2P0[1] * triDistP2[0]);

    distToLine = sqrt(POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2])) * tmpScalar;

    // factor -1 for triDistP2

    lineLambda = ((-triDistP2[0] * triAlongSideP2P0[0]) + (-triDistP2[1] * triAlongSideP2P0[1]) +
                  (-triDistP2[2] * triAlongSideP2P0[2])) *
                 POW2(tmpScalar);

    if (distToLine < M_MINDISTANCETOSIDELINE) {
        if (lineLambda >= 0. && lineLambda <= 1.) {
            distToLineMin = distToLine;
            correctionCounter++;
            correctionLineIndex = 2;
        } /* lambda */
    }     /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if (correctionCounter == 2) {
        const double upEps[3] = {P[0] + M_DISTANCECORRECTIONN3 * triN3[0],
                                 P[1] + M_DISTANCECORRECTIONN3 * triN3[1],
                                 P[2] + M_DISTANCECORRECTIONN3 * triN3[2]};
        const double downEps[3] = {P[0] - M_DISTANCECORRECTIONN3 * triN3[0],
                                   P[1] - M_DISTANCECORRECTIONN3 * triN3[1],
                                   P[2] - M_DISTANCECORRECTIONN3 * triN3[2]};

        // compute IqS term

        const double hUp = (triN3[0] * (upEps[0] - triCenter[0])) + (triN3[1] * (upEps[1] - triCenter[1])) +
                           (triN3[2] * (upEps[2] - triCenter[2]));

        const double solidAngleUp = SolidAngleTriangle(triData, upEps);

        const double hDown = (triN3[0] * (downEps[0] - triCenter[0])) + (triN3[1] * (downEps[1] - triCenter[1])) +
                             (triN3[2] * (downEps[2] - triCenter[2]));

        const double solidAngleDown = SolidAngleTriangle(triData, downEps);

        // compute IqL

        double IqLFieldAndPotentialUp[4];
        IqLFieldAndPotential(triData, upEps, 9, 9, 9, IqLFieldAndPotentialUp); /* no line correction */

        double IqLFieldAndPotentialDown[4];
        IqLFieldAndPotential(triData, downEps, 9, 9, 9, IqLFieldAndPotentialDown); /* no line correction */

        result[0] = M_ONEOVER_4PI_EPS0 * 0.5 *
                    ((triN3[0] * solidAngleUp + IqLFieldAndPotentialUp[0]) +
                     (triN3[0] * solidAngleDown + IqLFieldAndPotentialDown[0]));
        result[1] = M_ONEOVER_4PI_EPS0 * 0.5 *
                    ((triN3[1] * solidAngleUp + IqLFieldAndPotentialUp[1]) +
                     (triN3[1] * solidAngleDown + IqLFieldAndPotentialDown[1]));
        result[2] = M_ONEOVER_4PI_EPS0 * 0.5 *
                    ((triN3[2] * solidAngleUp + IqLFieldAndPotentialUp[2]) +
                     (triN3[2] * solidAngleDown + IqLFieldAndPotentialDown[2]));

        result[3] = M_ONEOVER_4PI_EPS0 * 0.5 *
                    ((-hUp * solidAngleUp - IqLFieldAndPotentialUp[3]) +
                     (-hDown * solidAngleDown - IqLFieldAndPotentialDown[3]));

        return;
    }

    const double h =
        (triN3[0] * (P[0] - triCenter[0])) + (triN3[1] * (P[1] - triCenter[1])) + (triN3[2] * (P[2] - triCenter[2]));

    const double triSolidAngle = SolidAngleTriangle(triData, P);

    double IqLFieldAndPhi[4];
    IqLFieldAndPotential(triData, P, correctionCounter, correctionLineIndex, distToLineMin, IqLFieldAndPhi);


    result[0] = M_ONEOVER_4PI_EPS0 * (triN3[0] * triSolidAngle + IqLFieldAndPhi[0]);
    result[1] = M_ONEOVER_4PI_EPS0 * (triN3[1] * triSolidAngle + IqLFieldAndPhi[1]);
    result[2] = M_ONEOVER_4PI_EPS0 * (triN3[2] * triSolidAngle + IqLFieldAndPhi[2]);
    result[3] = M_ONEOVER_4PI_EPS0 * ((-h * triSolidAngle) - IqLFieldAndPhi[3]);

    return;
}
