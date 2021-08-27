#include "KGExtrudedSurfaceMesher.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"

#include <algorithm>

//#define MERGE_DIST 1.e-4

namespace KGeoBag
{
void KGExtrudedSurfaceMesher::VisitWrappedSurface(KGWrappedSurface<KGExtrudedObject>* extrudedSurface)
{
    Discretize(extrudedSurface->GetObject().operator->());

    return;
}

void KGExtrudedSurfaceMesher::Discretize(KGExtrudedObject* object)
{
    fExtrudedObject = object;

    bool refineMesh = fExtrudedObject->RefineMesh();

    std::vector<unsigned int> nInnerCoords(fExtrudedObject->GetNInnerSegments(), 0);
    std::vector<unsigned int> nOuterCoords(fExtrudedObject->GetNOuterSegments(), 0);

    std::vector<unsigned int> innerDisc(fExtrudedObject->GetNInnerSegments(), 0);
    std::vector<unsigned int> outerDisc(fExtrudedObject->GetNOuterSegments(), 0);

    std::vector<std::vector<double>> innerCoords;
    std::vector<std::vector<double>> outerCoords;

    bool discretizationIsOk = false;

    bool discretizationHasChanged = false;
    unsigned int nDisc = fExtrudedObject->GetNDisc();
    int loopCounter = 0;

    while (!discretizationIsOk) {
        // compute the total outer length
        double totalOuterLength = 0.;
        for (unsigned int i = 0; i < fExtrudedObject->GetNOuterSegments(); i++)
            totalOuterLength += fExtrudedObject->GetOuterSegment(i)->GetLength();

        // compute the total inner length
        double totalInnerLength = 0.;
        for (unsigned int i = 0; i < fExtrudedObject->GetNInnerSegments(); i++)
            totalInnerLength += fExtrudedObject->GetInnerSegment(i)->GetLength();

        // First, we start with the outer segments

        // assign the segments a discretization according to their relative length
        int nAssignedOuterSegments = 0;
        for (unsigned int i = 0; i < fExtrudedObject->GetNOuterSegments(); i++) {
            int nSegments;
            if (totalOuterLength > totalOuterLength)
                nSegments = (int) (nDisc * totalOuterLength / totalInnerLength *
                                   fExtrudedObject->GetOuterSegment(i)->GetLength() / totalOuterLength);
            else
                nSegments = (int) (nDisc * fExtrudedObject->GetOuterSegment(i)->GetLength() / totalOuterLength);

            if (const auto* a = dynamic_cast<const KGExtrudedObject::Arc*>(fExtrudedObject->GetOuterSegment(i))) {
                if (nSegments < a->GetAngularSpread() / (M_PI / 6.))
                    nSegments = a->GetAngularSpread() / (M_PI / 6.);
                if (nSegments < 1)
                    nSegments = 1;
            }
            else if (nSegments < 4)
                nSegments = 4;
            outerDisc[i] = nSegments;
            nAssignedOuterSegments += nSegments;
        }

        if (nAssignedOuterSegments != (int) nDisc)
            discretizationHasChanged = true;

        // next, we add the inner segments

        // assign the segments a discretization according to their relative length
        int nAssignedInnerSegments = 0;
        for (unsigned int i = 0; i < fExtrudedObject->GetNInnerSegments(); i++) {
            int nSegments;
            if (totalInnerLength > totalOuterLength)
                nSegments = (int) (nDisc * totalInnerLength / totalOuterLength *
                                   fExtrudedObject->GetInnerSegment(i)->GetLength() / totalInnerLength);
            else
                nSegments = (int) (nDisc * fExtrudedObject->GetInnerSegment(i)->GetLength() / totalInnerLength);

            if (const auto* a = dynamic_cast<const KGExtrudedObject::Arc*>(fExtrudedObject->GetInnerSegment(i))) {
                if (nSegments < a->GetAngularSpread() / (M_PI / 6.))
                    nSegments = a->GetAngularSpread() / (M_PI / 6.);
                else if (fExtrudedObject->GetNInnerSegments() < 10 && fExtrudedObject->GetNOuterSegments() < 10)
                    nSegments += 7;
                if (nSegments < 1)
                    nSegments = 1;
            }
            else if (nSegments < 4)
                nSegments = 4;

            if (fExtrudedObject->GetNOuterSegments() > i)
                if (fExtrudedObject->GetInnerSegment(i)->GetNDisc() ==
                    fExtrudedObject->GetOuterSegment(i)->GetNDisc()) {
                    innerDisc[i] = fExtrudedObject->GetInnerSegment(i)->GetNDisc() + 1;
                }

            innerDisc[i] = nSegments;
            nAssignedInnerSegments += nSegments;
        }

        if (nAssignedInnerSegments != (int) nDisc)
            discretizationHasChanged = true;

        if (refineMesh && (nAssignedOuterSegments != nAssignedInnerSegments)) {
            nDisc += 1;
            continue;
        }

        discretizationIsOk = true;

        loopCounter++;

        if (nDisc < 8) {
            nDisc = loopCounter * 100;
        }

        if (loopCounter > 999) {
            // this probably needs to be rethought...
            std::stringstream s;
            s << "This extruded surface has undergone " << loopCounter
              << " iterations, and the boundaries have not yet been simultaneously discretized.  This algorithm probably needs to be made more general!";

            // KIOManager::GetInstance()->
            // 	Message("ExtrudedSurface","AddTo",s.str(),2);
        }
    }

    if (discretizationHasChanged) {
        std::stringstream s;
        s << "In order to properly match boundaries between the inner and outer loops of this surface, the variable ExtrudedSurface::fNDisc has been modified from "
          << fExtrudedObject->GetNDisc() << " to " << nDisc << ".";

        // KIOManager::GetInstance()->
        //   Message("ExtrudedSurface","AddTo",s.str(),0);
    }

    // add the inner segments to the electrode manager

    for (unsigned int i = 0; i < fExtrudedObject->GetNInnerSegments(); i++) {
        if (i != 0)
            innerCoords.pop_back();
        DiscretizeSegment(fExtrudedObject->GetInnerSegment(i), innerDisc.at(i), innerCoords, nInnerCoords[i]);

        if (fIsModifiable)
            ModifyInnerSegment(i, innerCoords);
    }

    // add the outer segments to the electrode manager
    for (unsigned int i = 0; i < fExtrudedObject->GetNOuterSegments(); i++) {
        if (i != 0)
            outerCoords.pop_back();
        DiscretizeSegment(fExtrudedObject->GetOuterSegment(i), outerDisc.at(i), outerCoords, nOuterCoords[i]);

        if (fIsModifiable)
            ModifyOuterSegment(i, outerCoords);
    }

    if (fIsModifiable)
        ModifySurface(innerCoords, outerCoords, nInnerCoords, nOuterCoords);

    DiscretizeEnclosedEnds(innerCoords, outerCoords, nDisc, fExtrudedObject->MeshMergeDistance());

    if (!(fExtrudedObject->ClosedLoops())) {
        DiscretizeLoopEnds();
    }
}

//____________________________________________________________________________

void KGExtrudedSurfaceMesher::DiscretizeSegment(const KGExtrudedObject::Line* line, const unsigned int nDisc,
                                                std::vector<std::vector<double>>& coords, unsigned int& counter)
{
    if (const auto* arc = dynamic_cast<const KGExtrudedObject::Arc*>(line))
        return DiscretizeSegment(arc, nDisc, coords, counter);

    double z_len = fExtrudedObject->GetZMax() - fExtrudedObject->GetZMin();

    double xy_len = line->GetLength();

    double n1[3] = {line->GetP2(0) - line->GetP1(0), line->GetP2(1) - line->GetP1(1), 0};

    double n2[3] = {0., 0., 1.};

    for (double& i : n1)
        i /= xy_len;

    double p0[3] = {line->GetP1(0), line->GetP1(1), fExtrudedObject->GetZMin()};

    double theta = fabs(atan((line->GetP2(0) - line->GetP1(0)) / (line->GetP2(1) - line->GetP1(1))));
    if (theta > 3. * M_PI / 2)
        theta = 2. * M_PI - theta;
    else if (theta > M_PI)
        theta -= M_PI;
    else if (theta > M_PI / 2.)
        theta = M_PI - theta;

    double theta0 = fabs(atan(line->GetP1(0) / line->GetP1(1)));
    if (theta0 > 3. * M_PI / 2)
        theta0 = 2. * M_PI - theta0;
    else if (theta0 > M_PI)
        theta0 -= M_PI;
    else if (theta0 > M_PI / 2.)
        theta0 = M_PI - theta0;

    int nDisc_xy = nDisc * sin(fabs(theta - theta0));
    if (nDisc_xy < 1)
        nDisc_xy = 1;

    std::vector<double> tmp(nDisc_xy, 0);
    DiscretizeInterval(xy_len, nDisc_xy, fExtrudedObject->GetDiscretizationPower(), tmp);
    std::vector<double> xy;
    xy.push_back(p0[0]);
    xy.push_back(p0[1]);
    coords.push_back(xy);
    counter++;
    xy[0] += n1[0] * xy_len;
    xy[1] += n1[1] * xy_len;
    coords.push_back(xy);
    counter++;

    auto* r = new KGMeshRectangle(xy_len, z_len, p0, n1, n2);

    RefineAndAddElement(r,
                        nDisc,
                        fExtrudedObject->GetDiscretizationPower(),
                        nDisc,
                        fExtrudedObject->GetExtrudedMeshPower());
}

//____________________________________________________________________________

void KGExtrudedSurfaceMesher::DiscretizeSegment(const KGExtrudedObject::Arc* arc, const unsigned int nDisc,
                                                std::vector<std::vector<double>>& coords, unsigned int& counter)
{
    double z_len = fExtrudedObject->GetZMax() - fExtrudedObject->GetZMin();

    int nDisc_xy = nDisc + 1;

    std::vector<double> dPhi(nDisc_xy, 0);

    DiscretizeInterval((arc->GetPhiEnd() - arc->GetPhiStart()), nDisc_xy - 1, 1, dPhi);
    dPhi[0] += arc->GetPhiStart();
    for (int i = 1; i < nDisc_xy; i++)
        dPhi[i] += dPhi[i - 1];
    for (int i = nDisc_xy - 1; i > 0; i--)
        dPhi[i] = dPhi[i - 1];
    dPhi[0] = arc->GetPhiStart();

    double n2[3] = {0., 0., 1.};

    double n1[3] = {};
    n1[2] = 0;

    double p0[3];
    p0[2] = fExtrudedObject->GetZMin();

    double xy_len = 0;

    for (int i = 0; i < nDisc_xy - 1; i++) {
        p0[0] = arc->GetCenter(0) + arc->GetRadius() * cos(dPhi[i]);
        p0[1] = arc->GetCenter(1) + arc->GetRadius() * sin(dPhi[i]);
        n1[0] = (arc->GetCenter(0) + arc->GetRadius() * cos(dPhi[i + 1])) - p0[0];
        n1[1] = (arc->GetCenter(1) + arc->GetRadius() * sin(dPhi[i + 1])) - p0[1];

        xy_len = sqrt(n1[0] * n1[0] + n1[1] * n1[1]);
        n1[0] /= xy_len;
        n1[1] /= xy_len;

        std::vector<double> xy;
        xy.push_back(p0[0]);
        xy.push_back(p0[1]);
        coords.push_back(xy);
        counter++;

        auto* r = new KGMeshRectangle(xy_len, z_len, p0, n1, n2);

        RefineAndAddElement(r,
                            1,
                            fExtrudedObject->GetDiscretizationPower(),
                            z_len / xy_len,
                            fExtrudedObject->GetExtrudedMeshPower());
    }

    std::vector<double> xy(2);
    // add the last point on the arc
    xy[0] = p0[0] + n1[0] * xy_len;
    xy[1] = p0[1] + n1[1] * xy_len;
    coords.push_back(xy);
    counter++;
}

//____________________________________________________________________________

void KGExtrudedSurfaceMesher::DiscretizeEnclosedEnds(std::vector<std::vector<double>>& iCoords,
                                                     std::vector<std::vector<double>>& oCoords, unsigned int nDisc,
                                                     double mergeDist)
{
    // For surfaces with closed ends, this method constructs the end caps.

    std::vector<std::vector<double>> augmentedInnerCoords = iCoords;
    std::vector<std::vector<double>> augmentedOuterCoords = oCoords;

    // We first cast the inner coordinates onto the outer boundary, and vice versa

    for (unsigned int i = 0; i < oCoords.size(); i++) {
        std::vector<double> P_int(2, 0);

        if (!fExtrudedObject->ClosedLoops()) {
            if (i == 0) {
                auto it = augmentedInnerCoords.begin();
                augmentedInnerCoords.insert(it, iCoords.at(0));
                continue;
            }
            else if (i == (oCoords.size() - 1)) {
                augmentedInnerCoords.push_back(iCoords.at(iCoords.size() - 1));
                continue;
            }
        }

        // since we restrict our boundaries to be convex, we are guaranteed that a
        // cast point will have exactly one corresponding representation on the
        // other boundary
        for (unsigned int j = 0; j < iCoords.size(); j++) {
            if (fExtrudedObject->RayIntersectsLineSeg(oCoords.at(i),
                                                      iCoords.at(j % iCoords.size()),
                                                      iCoords.at((j + 1) % iCoords.size()),
                                                      P_int)) {
                auto it = augmentedInnerCoords.begin() + i + ((j + 1) % iCoords.size());
                augmentedInnerCoords.insert(it, P_int);
                break;
            }
        }
    }

    for (unsigned int i = 0; i < iCoords.size(); i++) {
        std::vector<double> P_int(2, 0);

        if (!fExtrudedObject->ClosedLoops()) {
            if (i == 0) {
                auto it = augmentedOuterCoords.begin();
                augmentedOuterCoords.insert(it, oCoords.at(0));
                continue;
            }
            else if (i == iCoords.size() - 1) {
                augmentedOuterCoords.push_back(oCoords.at(oCoords.size() - 1));
                continue;
            }
        }

        // since we restrict our boundaries to be convex, we are guaranteed that a
        // cast point will have exactly one corresponding representation on the
        // other boundary
        for (unsigned int j = 0; j < oCoords.size(); j++) {
            if (fExtrudedObject->RayIntersectsLineSeg(iCoords.at(i),
                                                      oCoords.at(j % oCoords.size()),
                                                      oCoords.at((j + 1) % oCoords.size()),
                                                      P_int)) {
                auto it = augmentedOuterCoords.begin() + i + ((j + 1) % oCoords.size());
                augmentedOuterCoords.insert(it, P_int);
                break;
            }
        }
    }

    /////////////////////
    // now we try to merge points
    for (unsigned int i = 0; i < augmentedOuterCoords.size(); i++) {
        // we attempt to merge points i and i+1
        double x0, x1, x2, x3, y0, y1, y2, y3;
        x1 = augmentedOuterCoords.at(i).at(0);
        y1 = augmentedOuterCoords.at(i).at(1);
        x2 = augmentedOuterCoords.at((i + 1) % augmentedOuterCoords.size()).at(0);
        y2 = augmentedOuterCoords.at((i + 1) % augmentedOuterCoords.size()).at(1);

        double dist = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

        // if the points aren't close, we shouldn't try to merge them
        if (dist > mergeDist)
            continue;

        int pointsAreOnALine = 0;  // 0 = false; 1 = below; 2 = above; 3 = both;

        x0 = augmentedOuterCoords.at((i - 1 + augmentedOuterCoords.size()) % augmentedOuterCoords.size()).at(0);
        y0 = augmentedOuterCoords.at((i - 1 + augmentedOuterCoords.size()) % augmentedOuterCoords.size()).at(1);
        x3 = augmentedOuterCoords.at((i + 2) % augmentedOuterCoords.size()).at(0);
        y3 = augmentedOuterCoords.at((i + 2) % augmentedOuterCoords.size()).at(1);

        double u = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / (dist * dist);

        if (fabs((x1 + u * (x2 - x1)) - x0) < 1.e-10 && fabs((y1 + u * (y2 - y1)) - y0) < 1.e-10)
            pointsAreOnALine += 1;

        u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / (dist * dist);

        if (fabs((x1 + u * (x2 - x1)) - x3) < 1.e-10 && fabs((y1 + u * (y2 - y1)) - y3) < 1.e-10)
            pointsAreOnALine += 2;

        // if the points do not lie on a common line segment, we shouldn't try to
        // merge them
        if (pointsAreOnALine == 0)
            continue;

        x0 = augmentedInnerCoords.at(i).at(0);
        y0 = augmentedInnerCoords.at(i).at(1);
        x3 = augmentedInnerCoords.at((i + 1) % augmentedOuterCoords.size()).at(0);
        y3 = augmentedInnerCoords.at((i + 1) % augmentedOuterCoords.size()).at(1);

        double dist1, dist2;
        dist1 = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        dist2 = sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));

        if (pointsAreOnALine == 2 || (pointsAreOnALine == 3 && dist1 < dist2))
        // if ( pointsAreOnALine == 2)
        {
            augmentedOuterCoords.erase(augmentedOuterCoords.begin() + i);
            augmentedInnerCoords.erase(augmentedInnerCoords.begin() + i);
            i--;
        }
        else if (pointsAreOnALine == 1 || (pointsAreOnALine == 3 && dist2 < dist1))
        // else if (pointsAreOnALine == 1)
        {
            augmentedOuterCoords.erase(augmentedOuterCoords.begin() + (i + 1) % augmentedOuterCoords.size());
            augmentedInnerCoords.erase(augmentedInnerCoords.begin() + (i + 1) % augmentedOuterCoords.size());
        }
    }

    for (unsigned int i = 0; i < augmentedInnerCoords.size(); i++) {
        // we attempt to merge points i and i+1
        double x0, x1, x2, x3, y0, y1, y2, y3;
        x1 = augmentedInnerCoords.at(i).at(0);
        y1 = augmentedInnerCoords.at(i).at(1);
        x2 = augmentedInnerCoords.at((i + 1) % augmentedInnerCoords.size()).at(0);
        y2 = augmentedInnerCoords.at((i + 1) % augmentedInnerCoords.size()).at(1);

        double dist = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));

        // if the points aren't close, we shouldn't try to merge them
        if (dist > mergeDist)
            continue;

        int pointsAreOnALine = 0;  // 0 = false; 1 = below; 2 = above; 3 = both;

        x0 = augmentedInnerCoords.at((i - 1 + augmentedInnerCoords.size()) % augmentedInnerCoords.size()).at(0);
        y0 = augmentedInnerCoords.at((i - 1 + augmentedInnerCoords.size()) % augmentedInnerCoords.size()).at(1);
        x3 = augmentedInnerCoords.at((i + 2) % augmentedInnerCoords.size()).at(0);
        y3 = augmentedInnerCoords.at((i + 2) % augmentedInnerCoords.size()).at(1);

        double u = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / (dist * dist);

        if (fabs((x1 + u * (x2 - x1)) - x0) < 1.e-10 && fabs((y1 + u * (y2 - y1)) - y0) < 1.e-10)
            pointsAreOnALine += 1;

        u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / (dist * dist);

        if (fabs((x1 + u * (x2 - x1)) - x3) < 1.e-10 && fabs((y1 + u * (y2 - y1)) - y3) < 1.e-10)
            pointsAreOnALine += 2;

        // if the points do not lie on a common line segment, we shouldn't try to
        // merge them
        if (pointsAreOnALine == 0)
            continue;

        x0 = augmentedOuterCoords.at(i).at(0);
        y0 = augmentedOuterCoords.at(i).at(1);
        x3 = augmentedOuterCoords.at((i + 1) % augmentedOuterCoords.size()).at(0);
        y3 = augmentedOuterCoords.at((i + 1) % augmentedOuterCoords.size()).at(1);

        double dist1, dist2;
        dist1 = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
        dist2 = sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));

        if (pointsAreOnALine == 2 || (pointsAreOnALine == 3 && dist1 < dist2))
        // if (pointsAreOnALine == 2)
        {
            augmentedOuterCoords.erase(augmentedOuterCoords.begin() + i);
            augmentedInnerCoords.erase(augmentedInnerCoords.begin() + i);
            i--;
        }
        else if (pointsAreOnALine == 1 || (pointsAreOnALine == 3 && dist2 < dist1))
        // else if (pointsAreOnALine == 1)
        {
            augmentedOuterCoords.erase(augmentedOuterCoords.begin() + (i + 1) % augmentedInnerCoords.size());
            augmentedInnerCoords.erase(augmentedInnerCoords.begin() + (i + 1) % augmentedInnerCoords.size());
        }
    }
    /////////////////////////

    /////////////////////////
    /*
         // now we try to merge points (simplified algorithm)
         for (unsigned int i=0;i<augmentedOuterCoords.size();i++)
         {
         // we attempt to merge points i and i+1
         double x0,x1,y0,y1;
         x0 = augmentedOuterCoords.at(i).at(0);
         y0 = augmentedOuterCoords.at(i).at(1);
         x1 = augmentedOuterCoords.at((i+1)%augmentedOuterCoords.size()).at(0);
         y1 = augmentedOuterCoords.at((i+1)%augmentedOuterCoords.size()).at(1);

         double dist = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));

         // if the points aren't close, we shouldn't try to merge them
         if (dist<mergeDist)
         {
         augmentedOuterCoords.erase(augmentedOuterCoords.begin()+i);
         i++;
         }
         }

         for (unsigned int i=0;i<augmentedInnerCoords.size();i++)
         {
         // we attempt to merge points i and i+1
         double x0,x1,y0,y1;
         x0 = augmentedInnerCoords.at(i).at(0);
         y0 = augmentedInnerCoords.at(i).at(1);
         x1 = augmentedInnerCoords.at((i+1)%augmentedInnerCoords.size()).at(0);
         y1 = augmentedInnerCoords.at((i+1)%augmentedInnerCoords.size()).at(1);

         double dist = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));

         // if the points aren't close, we shouldn't try to merge them
         if (dist<mergeDist)
         {
         augmentedInnerCoords.erase(augmentedInnerCoords.begin()+i);
         i++;
         }
         }
         */
    /////////////////////////

    // if we have closed loops, we can sort the points (and, thus, maintain
    // backwards compatibility with the detector region's flange!)
    // if (fExtrudedObject->ClosedLoops())
    if (true) {
        if (fExtrudedObject->IsBackwards()) {
            std::sort(augmentedInnerCoords.rbegin(), augmentedInnerCoords.rend(), KGExtrudedObject::CompareTheta);
            std::sort(augmentedOuterCoords.rbegin(), augmentedOuterCoords.rend(), KGExtrudedObject::CompareTheta);
        }
        else {
            std::sort(augmentedInnerCoords.begin(), augmentedInnerCoords.end(), KGExtrudedObject::CompareTheta);
            std::sort(augmentedOuterCoords.begin(), augmentedOuterCoords.end(), KGExtrudedObject::CompareTheta);
        }
    }

    iCoords.clear();
    oCoords.clear();

    iCoords = augmentedInnerCoords;
    oCoords = augmentedOuterCoords;

    unsigned int nStrips = iCoords.size();
    if (!fExtrudedObject->ClosedLoops())
        nStrips--;

    int inner_low_index = 0;
    int outer_low_index = 0;
    int inner_high_index = 1;
    int outer_high_index = 1;

    // we now break the regions into trapezoidal strips
    for (unsigned int i = 0; i < nStrips; i++) {
        inner_low_index = (i) % iCoords.size();
        outer_low_index = (i) % oCoords.size();

        inner_high_index = (i + 1) % iCoords.size();
        outer_high_index = (i + 1) % oCoords.size();

        double inner_low[3], inner_high[3], outer_low[3], outer_high[3];
        inner_low[0] = iCoords.at(inner_low_index).at(0);
        inner_low[1] = iCoords.at(inner_low_index).at(1);
        inner_high[0] = iCoords.at(inner_high_index).at(0);
        inner_high[1] = iCoords.at(inner_high_index).at(1);
        outer_low[0] = oCoords.at(outer_low_index).at(0);
        outer_low[1] = oCoords.at(outer_low_index).at(1);
        outer_high[0] = oCoords.at(outer_high_index).at(0);
        outer_high[1] = oCoords.at(outer_high_index).at(1);

        // for points whose theta are the same, we can just skip them and go on to
        // the next step
        double theta_low, theta_high;
        theta_low = KGExtrudedObject::Theta(inner_low[0], inner_low[1]);
        theta_high = KGExtrudedObject::Theta(inner_high[0], inner_high[1]);
        if (fabs(theta_low - theta_high) < 1.e-6)
            continue;
        theta_low = KGExtrudedObject::Theta(outer_low[0], outer_low[1]);
        theta_high = KGExtrudedObject::Theta(outer_high[0], outer_high[1]);
        if (fabs(theta_low - theta_high) < 1.e-6)
            continue;

        double low_unit[2] = {outer_low[0] - inner_low[0], outer_low[1] - inner_low[1]};

        double high_unit[2] = {outer_high[0] - inner_high[0], outer_high[1] - inner_high[1]};

        double low_dist = sqrt((outer_low[0] - inner_low[0]) * (outer_low[0] - inner_low[0]) +
                               (outer_low[1] - inner_low[1]) * (outer_low[1] - inner_low[1]));

        double high_dist = sqrt((outer_high[0] - inner_high[0]) * (outer_high[0] - inner_high[0]) +
                                (outer_high[1] - inner_high[1]) * (outer_high[1] - inner_high[1]));

        for (unsigned int j = 0; j < 2; j++) {
            low_unit[j] /= low_dist;
            high_unit[j] /= high_dist;
        }

        double apprxRadLow = fabs(sqrt(outer_low[0] * outer_low[0] + outer_low[1] * outer_low[1]) -
                                  sqrt(inner_low[0] * inner_low[0] + inner_low[1] * inner_low[1]));
        double apprxRadHigh = fabs(sqrt(outer_high[0] * outer_high[0] + outer_high[1] * outer_high[1]) -
                                   sqrt(inner_high[0] * inner_high[0] + inner_high[1] * inner_high[1]));
        double apprxRad = (apprxRadLow < apprxRadHigh ? apprxRadLow : apprxRadHigh);
        int n_disc =
            6. * nDisc * apprxRad / (2. * M_PI * sqrt(outer_low[0] * outer_low[0] + outer_low[1] * outer_low[1]));

        if (n_disc < 3)
            n_disc += 3;

        std::vector<double> tmp_low(n_disc, 0);
        DiscretizeInterval(low_dist,
                           n_disc,
                           // GetDiscretizationPower(),
                           1.3,
                           tmp_low);
        std::vector<double> tmp_high(n_disc, 0);
        DiscretizeInterval(high_dist,
                           n_disc,
                           // GetDiscretizationPower(),
                           1.3,
                           tmp_high);

        double i_l[2] = {inner_low[0], inner_low[1]};
        double i_h[2] = {inner_high[0], inner_high[1]};
        double o_l[2], o_h[2];
        for (int j = 0; j < 2; j++) {
            o_l[j] = i_l[j] + low_unit[j] * tmp_low[0];
            o_h[j] = i_h[j] + high_unit[j] * tmp_high[0];
        }

        for (int j = 0; j < n_disc; j++) {
            double p0[3], p1[3], p2[3];
            KGMeshTriangle* t = nullptr;

            p0[2] = p1[2] = p2[2] = fExtrudedObject->GetZMin();
            p0[0] = i_l[0];
            p0[1] = i_l[1];
            p1[0] = i_h[0];
            p1[1] = i_h[1];
            p2[0] = o_h[0];
            p2[1] = o_h[1];

            t = new KGMeshTriangle(p0, p1, p2);
            AddElement(t);

            p0[2] = p1[2] = p2[2] = fExtrudedObject->GetZMax();
            t = new KGMeshTriangle(p0, p1, p2);
            AddElement(t);

            if ((fabs(o_l[0] - o_h[0]) < 1.e-12) && (fabs(o_l[1] - o_h[1]) < 1.e-12))
                continue;
            else {
                p0[2] = p1[2] = p2[2] = fExtrudedObject->GetZMin();
                p1[0] = o_l[0];
                p1[1] = o_l[1];
                t = new KGMeshTriangle(p0, p1, p2);
                AddElement(t);

                p0[2] = p1[2] = p2[2] = fExtrudedObject->GetZMax();
                t = new KGMeshTriangle(p0, p1, p2);
                AddElement(t);
            }

            for (int k = 0; k < 2; k++) {
                i_l[k] = o_l[k];
                i_h[k] = o_h[k];
                if (j < n_disc - 1) {
                    o_l[k] += low_unit[k] * tmp_low[j + 1];
                    o_h[k] += high_unit[k] * tmp_high[j + 1];
                }
            }
        }
    }
}

//____________________________________________________________________________

void KGExtrudedSurfaceMesher::DiscretizeLoopEnds() {}
}  // namespace KGeoBag
