#include "KGRotatedSurfaceMesher.hh"

#include "KGMeshTriangle.hh"

namespace KGeoBag
{
void KGRotatedSurfaceMesher::VisitWrappedSurface(KGWrappedSurface<KGRotatedObject>* rotatedSurface)
{
    std::shared_ptr<KGRotatedObject> rotatedObject = rotatedSurface->GetObject();

    double length = 0.;

    for (unsigned int i = 0; i < rotatedObject->GetNSegments(); i++)
        length += rotatedObject->GetSegment(i)->GetLength();

    double tmp_len = 0;
    unsigned int nPolyBegin_seg = rotatedObject->GetNPolyBegin();
    unsigned int nPolyEnd_seg = 0;

    if (rotatedObject->GetNSegments() == 1)
        nPolyEnd_seg = rotatedObject->GetNPolyEnd();
    else {
        for (unsigned int i = 1; i < rotatedObject->GetNSegments(); i++) {
            tmp_len += rotatedObject->GetSegment(i - 1)->GetLength();
            double ratio = tmp_len / length;
            nPolyEnd_seg = ceil((1. - ratio) * rotatedObject->GetNPolyBegin() + (ratio) *rotatedObject->GetNPolyEnd());
            DiscretizeSegment(rotatedObject->GetSegment(i - 1), nPolyBegin_seg, nPolyEnd_seg);
            nPolyBegin_seg = nPolyEnd_seg;
        }
    }
    DiscretizeSegment(rotatedObject->GetSegment(rotatedObject->GetNSegments() - 1), nPolyBegin_seg, nPolyEnd_seg);
}

//____________________________________________________________________________

void KGRotatedSurfaceMesher::DiscretizeSegment(const KGRotatedObject::Line* line, const unsigned int nPolyBegin,
                                               const unsigned int nPolyEnd)
{
    if (const auto* arc = dynamic_cast<const KGRotatedObject::Arc*>(line))
        return DiscretizeSegment(arc, nPolyBegin, nPolyEnd);

    // First, we find the length of a side of the polygon in the middle of the
    // segment
    double rMid = fabs(line->GetP1(1) + line->GetP2(1)) * .5;
    int nPolyMid = .5 * (nPolyBegin + nPolyEnd);
    double lenSeg = 2. * rMid * sin(M_PI / nPolyMid);

    // We then use this value to determine the # of discretizations along the
    // segment

    int nDisc = ceil(line->GetLength() / lenSeg) + 1;

    // unit: a unit vector pointing from the start point to the end point
    double unit[2] = {(line->GetP2(0) - line->GetP1(0)) / line->GetLength(),
                      (line->GetP2(1) - line->GetP1(1)) / line->GetLength()};

    // dlen: an incremental length from 0. to GetLength()
    std::vector<double> dlen(nDisc, 0);

    // alpha: incrementally increases from 0. to 1.
    std::vector<double> alpha(nDisc, 0);

    // nPoly: incrementally goes from nPolyBegin to nPolyEnd
    std::vector<int> nPoly(nDisc, 0);

    DiscretizeInterval(line->GetLength(), nDisc - 1, 2., dlen);
    DiscretizeInterval(1., nDisc - 1, 1, alpha);

    // By adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval.
    for (int i = 1; i < nDisc; i++) {
        dlen[i] += dlen[i - 1];
        alpha[i] += alpha[i - 1];
    }

    // We modify alpha to go from 0. to 1.
    for (int i = nDisc - 1; i > 0; i--) {
        dlen[i] = dlen[i - 1];
        alpha[i] = alpha[i - 1];
    }
    dlen[0] = 0.;
    alpha[0] = 0.;

    // We use alpha to figure out the nPoly array
    for (int i = 0; i < nDisc; i++)
        nPoly[i] = lround(nPolyEnd * alpha[i] + nPolyBegin * (1. - alpha[i]));

    if (nDisc > 3) {
        nPoly[1] = nPoly[0] = nPolyBegin;
        nPoly[nDisc - 2] = nPoly[nDisc - 1] = nPolyEnd;
    }

    // Now, we create our triangles

    KGMeshTriangle* t;
    double p0[3];
    double p1[3];
    double p2[3];

    int nThetaMax = (nPolyBegin > nPolyEnd) ? nPolyBegin : nPolyEnd;

    double r = 0;
    double r_last = line->GetP1(1);

    std::vector<double> theta(nThetaMax, 0);
    std::vector<double> theta_last(nThetaMax, 0);

    for (int i = 0; i < nPoly[0]; i++)
        theta_last[i] = i * (2. * M_PI / nPoly[0]);

    for (int i = 1; i < nDisc; i++) {
        // First, we compute our theta arrays for this pass
        for (int j = 0; j < nPoly[i]; j++)
            theta[j] = j * (2. * M_PI / nPoly[i]);

        // We traverse along the unit vector a distance dlen[i]
        r = line->GetP1(1) + unit[1] * dlen[i];

        // We first construct the triangles whose edges correspond to theta_last
        p0[2] = p2[2] = line->GetP1(0) + unit[0] * dlen[i - 1];
        p1[2] = line->GetP1(0) + unit[0] * dlen[i];

        if (r_last > 0.) {
            for (int j = 0; j < nPoly[i - 1]; j++) {
                p0[0] = r_last * cos(theta_last[j]);
                p0[1] = r_last * sin(theta_last[j]);

                p2[0] = r_last * cos(theta_last[(j + 1) % nPoly[i - 1]]);
                p2[1] = r_last * sin(theta_last[(j + 1) % nPoly[i - 1]]);

                int th_index = 0;
                while (theta_last[j] >= theta[th_index]) {
                    th_index++;
                    if (th_index == nPoly[i])
                        break;
                }

                if (j == nPoly[i - 1] - 1)
                    th_index = nPoly[i] - 1;

                p1[0] = r * cos(theta[th_index % nPoly[i]]);
                p1[1] = r * sin(theta[th_index % nPoly[i]]);

                t = new KGMeshTriangle(p0, p2, p1);
                AddElement(t);
            }
        }

        // We then construct the triangles whose edges correspond to theta
        p0[2] = p2[2] = line->GetP1(0) + unit[0] * dlen[i];
        p1[2] = line->GetP1(0) + unit[0] * dlen[i - 1];

        if (r > 0) {
            for (int j = 0; j < nPoly[i]; j++) {
                p0[0] = r * cos(theta[j]);
                p0[1] = r * sin(theta[j]);

                p2[0] = r * cos(theta[(j + 1) % nPoly[i]]);
                p2[1] = r * sin(theta[(j + 1) % nPoly[i]]);

                int th_index = 0;
                // while (theta_last[th_index]<theta[(j+1)%nPoly[i]])
                if (j == 0 || j == nPoly[i] - 1)
                    th_index = 0;
                else {
                    while (theta_last[th_index] < theta[(j) % nPoly[i]]) {
                        th_index++;
                        if (th_index == nPoly[i - 1]) {
                            th_index = nPoly[i - 1] - 1;
                            break;
                        }
                    }
                }

                p1[0] = r_last * cos(theta_last[th_index % nPoly[i - 1]]);
                p1[1] = r_last * sin(theta_last[th_index % nPoly[i - 1]]);

                t = new KGMeshTriangle(p0, p1, p2);
                AddElement(t);
            }
        }

        // Finally, we set our variables for the next iteration
        for (int j = 0; j < nPoly[i]; j++)
            theta_last[j] = theta[j];
        r_last = r;
    }
}

//____________________________________________________________________________

void KGRotatedSurfaceMesher::DiscretizeSegment(const KGRotatedObject::Arc* arc, const unsigned int nPolyBegin,
                                               const unsigned int nPolyEnd)
{
    double rMid = fabs(arc->GetCenter(1) + arc->GetRadius() * sin(arc->GetPhiMid()));
    int nPolyMid = .5 * (nPolyBegin + nPolyEnd);
    double lenSeg = 2. * rMid * sin(M_PI / nPolyMid);

    // We then use this value to determine the # of discretizations along the
    // segment

    int nDisc = ceil(arc->GetLength() / lenSeg) + 1;
    double deltaPhi = arc->GetPhiEnd() - arc->GetPhiStart();

    std::vector<int> nPoly(nDisc, 0);
    for (int i = 0; i < nDisc; i++)
        nPoly[i] = lround(nPolyEnd * i / (nDisc - 1.) + nPolyBegin * (1. - i / (nDisc - 1.)));

    // Now, we create our triangles

    KGMeshTriangle* t;
    double p0[3];
    double p1[3];
    double p2[3];

    int nThetaMax = nPolyBegin > nPolyEnd ? nPolyBegin : nPolyEnd;

    double r = 0;
    double r_last = arc->GetP1(1);

    std::vector<double> theta(nThetaMax, 0);
    std::vector<double> theta_last(nThetaMax, 0);

    for (int i = 0; i < nPoly[0]; i++)
        theta_last[i] = i * (2. * M_PI / nPoly[0]);

    for (int i = 1; i < nDisc; i++) {
        // First, we compute our theta arrays for this pass
        for (int j = 0; j < nPoly[i]; j++)
            theta[j] = j * (2. * M_PI / nPoly[i]);

        // We traverse along the arc a distance i/(nDisc-1.)*deltaPhi

        r = arc->GetCenter(1) + arc->GetRadius() * sin(arc->GetPhiStart() + i / (nDisc - 1.) * deltaPhi);

        // We first construct the triangles whose edges correspond to theta_last
        p0[2] = p2[2] =
            arc->GetCenter(0) + arc->GetRadius() * cos(arc->GetPhiStart() + (i - 1) / (nDisc - 1.) * deltaPhi);
        p1[2] = arc->GetCenter(0) + arc->GetRadius() * cos(arc->GetPhiStart() + (i) / (nDisc - 1.) * deltaPhi);

        for (int j = 0; j < nPoly[i - 1]; j++) {
            p0[0] = r_last * cos(theta_last[j]);
            p0[1] = r_last * sin(theta_last[j]);

            p2[0] = r_last * cos(theta_last[(j + 1) % nPoly[i - 1]]);
            p2[1] = r_last * sin(theta_last[(j + 1) % nPoly[i - 1]]);

            int th_index = 0;
            while (theta_last[j] >= theta[th_index]) {
                th_index++;
                if (th_index == nPoly[i])
                    break;
            }
            th_index--;
            if (th_index < 0)
                th_index = nPoly[i] - 1;

            p1[0] = r * cos(theta[th_index % nPoly[i]]);
            p1[1] = r * sin(theta[th_index % nPoly[i]]);

            t = new KGMeshTriangle(p0, p2, p1);
            AddElement(t);
        }

        // We then construct the triangles whose edges correspond to theta
        p0[2] = p2[2] = arc->GetCenter(0) + arc->GetRadius() * cos(arc->GetPhiStart() + (i) / (nDisc - 1.) * deltaPhi);
        p1[2] = arc->GetCenter(0) + arc->GetRadius() * cos(arc->GetPhiStart() + (i - 1) / (nDisc - 1.) * deltaPhi);

        for (int j = 0; j < nPoly[i]; j++) {
            p0[0] = r * cos(theta[j]);
            p0[1] = r * sin(theta[j]);

            p2[0] = r * cos(theta[(j + 1) % nPoly[i]]);
            p2[1] = r * sin(theta[(j + 1) % nPoly[i]]);

            int th_index = 0;
            while ((theta_last[th_index] < theta[(j + 1) % nPoly[i]]) && (th_index < nThetaMax))
                th_index++;

            p1[0] = r_last * cos(theta_last[th_index % nPoly[i - 1]]);
            p1[1] = r_last * sin(theta_last[th_index % nPoly[i - 1]]);

            t = new KGMeshTriangle(p0, p1, p2);
            AddElement(t);
        }

        // Finally, we set our variables for the next iteration
        for (int j = 0; j < nPoly[i]; j++)
            theta_last[j] = theta[j];
        r_last = r;
    }
}
}  // namespace KGeoBag
