#include "KGPortHousingSurfaceMesher.hh"

#include "KGLinearCongruentialGenerator.hh"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"

#include <algorithm>
#include <iomanip>

namespace KGeoBag
{
void KGPortHousingSurfaceMesher::VisitWrappedSurface(KGWrappedSurface<KGPortHousing>* portHousingSurface)
{
    fPortHousing = portHousingSurface->GetObject();

    std::vector<double> theta(fPortHousing->GetNPorts(), 0);
    std::vector<double> phi(fPortHousing->GetNPorts(), 0);
    std::vector<double> mid(fPortHousing->GetNPorts(), 0);
    std::vector<double> width(fPortHousing->GetNPorts(), 0);

    ComputeEnclosingBoxLengths(theta, phi, mid, width);

    // the ports are complicated, so they compute themselves.
    auto* rectangularPortDiscretizer = new KGPortHousingSurfaceMesher::RectangularPortDiscretizer(this);
    auto* circularPortDiscretizer = new KGPortHousingSurfaceMesher::CircularPortDiscretizer(this);
    for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {
        if (const auto* r = dynamic_cast<const KGPortHousing::RectangularPort*>(fPortHousing->GetPort(i)))
            rectangularPortDiscretizer->DiscretizePort(r);
        else if (const auto* c = dynamic_cast<const KGPortHousing::CircularPort*>(fPortHousing->GetPort(i)))
            circularPortDiscretizer->DiscretizePort(c);
    }
    delete rectangularPortDiscretizer;
    delete circularPortDiscretizer;

    // all that is left is to fill in the gaps around the ports.  First, we find
    // the areas that are covered by the ports.  This will be done w.r.t. the
    // local frame of the first port.

    double cen[3];
    double x_loc[3];
    double y_loc[3];
    double z_loc[3];
    double x_min;
    double x_max;

    KGCoordinateTransform* coordTransform = nullptr;

    for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {
        const KGPortHousing::Port* v = fPortHousing->GetPort(i);
        v->ComputeLocalFrame(cen, x_loc, y_loc, z_loc);

        if (i == 0) {
            coordTransform = new KGCoordinateTransform(cen, x_loc, y_loc, z_loc);
            double tmp[3];
            coordTransform->ConvertToLocalCoords(fPortHousing->GetAMain(), tmp, false);
            x_min = tmp[0];
            coordTransform->ConvertToLocalCoords(fPortHousing->GetBMain(), tmp, false);
            x_max = tmp[0];

            theta[i] = 0.;
            mid[i] = 0.;
        }
        else {
            double tmp[3];
            coordTransform->ConvertToLocalCoords(cen, tmp, false);
            mid[i] = tmp[0];
            coordTransform->ConvertToLocalCoords(z_loc, tmp, false);
            theta[i] = acos(tmp[2]);
            coordTransform->ConvertToLocalCoords(y_loc, tmp, false);
            if (tmp[2] < 0)
                theta[i] = 2. * M_PI - theta[i];
        }

        width[i] = v->GetBoxWidth();

        phi[i] = acos(1. - v->GetBoxLength() * v->GetBoxLength() /
                               (2. * fPortHousing->GetRMain() * fPortHousing->GetRMain()));
    }

    // now that we have the parameters of all of the port boxes, we fill in the
    // gaps

    double th = 0;
    double th_last = 0;

    double p0_loc[3];
    double p1_loc[3];
    double p2_loc[3];
    double p0[3];
    double p1[3];
    double p2[3];
    double n1[3];
    double n2[3];

    double a = fPortHousing->GetRMain() * sqrt(2. * (1. - cos(2. * M_PI / fPortHousing->GetPolyMain())));

    std::vector<double> dx(fPortHousing->GetNumDiscMain(), 0);

    for (int i = 1; i <= fPortHousing->GetPolyMain(); i++) {
        th = ((double) (i % fPortHousing->GetPolyMain())) / fPortHousing->GetPolyMain() * 2. * M_PI;

        p0_loc[2] = p2_loc[2] = fPortHousing->GetRMain() * cos(th_last);
        p0_loc[1] = p2_loc[1] = -fPortHousing->GetRMain() * sin(th_last);
        p1_loc[2] = fPortHousing->GetRMain() * cos(th);
        p1_loc[1] = -fPortHousing->GetRMain() * sin(th);

        std::vector<double> interval_start;
        std::vector<double> interval_end;

        interval_start.push_back(x_min);
        for (unsigned int j = 0; j < fPortHousing->GetNPorts(); j++) {
            if (ChordsIntersect(th_last, th, theta[j] - phi[j] / 2., theta[j] + phi[j] / 2.)) {
                interval_end.push_back(mid[j] - width[j] / 2.);
                interval_start.push_back(mid[j] + width[j] / 2.);
            }
        }
        interval_end.push_back(x_max);

        std::sort(interval_start.begin(), interval_start.end());
        std::sort(interval_end.begin(), interval_end.end());

        for (unsigned int j = 0; j < interval_start.size(); j++) {
            if (fabs(interval_start[j] - interval_end[j]) < 1.e-10)
                continue;

            int nInc = fPortHousing->GetNumDiscMain() * (interval_end[j] - interval_start[j]) / (x_max - x_min);

            // account for small gaps between ports
            if (nInc == 0)
                nInc++;

            DiscretizeInterval(interval_end[j] - interval_start[j], nInc, 2., dx);

            p0_loc[0] = p1_loc[0] = interval_start[j];
            for (int k = 0; k < nInc; k++) {
                p2_loc[0] = p0_loc[0] + dx[k];

                coordTransform->ConvertToGlobalCoords(p0_loc, p0, false);
                coordTransform->ConvertToGlobalCoords(p1_loc, p1, false);
                coordTransform->ConvertToGlobalCoords(p2_loc, p2, false);

                double tmp1 = 0;
                double tmp2 = 0;
                for (int m = 0; m < 3; m++) {
                    n1[m] = p1[m] - p0[m];
                    n2[m] = p2[m] - p0[m];
                    tmp1 += n1[m] * n1[m];
                    tmp2 += n2[m] * n2[m];
                }
                tmp1 = sqrt(tmp1);
                tmp2 = sqrt(tmp2);
                for (int m = 0; m < 3; m++) {
                    n1[m] /= tmp1;
                    n2[m] /= tmp2;
                }

                KGMeshRectangle* r = new KGMeshRectangle(a, p2_loc[0] - p0_loc[0], p0, n1, n2);
                AddElement(r);

                p0_loc[0] = p1_loc[0] = p2_loc[0];
            }
        }

        th_last = th;
    }

    delete coordTransform;
}

//____________________________________________________________________________

void KGPortHousingSurfaceMesher::ComputeEnclosingBoxLengths(std::vector<double>& theta, std::vector<double>& phi,
                                                            std::vector<double>& mid, std::vector<double>& width)
{
    // This function computes the lengths of a sides of the boxes that enclose
    // the special discretization around the port holes.  Special care is made
    // to ensure that boundaries are matched between the main cylinder and
    // subordinate shapes.
    //
    // theta[i]:  the angle of the mid-points of box i
    // phi[i]:    the angle subtended by box i
    // mid[i]:    the middle of box i
    // width[i]: the length of box i along the cylinder axis

    // the edges of the main cylinder in local coordinates
    double x_min = 0;
    double x_max = 0;

    // First we compute theta for each of the ports
    KGCoordinateTransform* coordTransform = nullptr;

    for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {
        // we grab the local frame for port <i>
        const KGPortHousing::Port* v = fPortHousing->GetPort(i);
        double cen[3];
        double x_loc[3];
        double y_loc[3];
        double z_loc[3];
        v->ComputeLocalFrame(cen, x_loc, y_loc, z_loc);

        if (i == 0) {
            // if it's the first port, then we use its local coordinate frame as the
            // primary coordinate frame
            coordTransform = new KGCoordinateTransform(cen, x_loc, y_loc, z_loc);
            double tmp[3];
            coordTransform->ConvertToLocalCoords(fPortHousing->GetAMain(), tmp, false);
            x_min = tmp[0];
            coordTransform->ConvertToLocalCoords(fPortHousing->GetBMain(), tmp, false);
            x_max = tmp[0];

            theta[i] = 0.;
            mid[i] = 0.;
        }
        else {
            // otherwise, we figure out theta with respect to the primary coordinate
            // frame
            double tmp[3];
            coordTransform->ConvertToLocalCoords(cen, tmp, false);
            mid[i] = tmp[0];
            coordTransform->ConvertToLocalCoords(z_loc, tmp, false);
            theta[i] = acos(tmp[2]);
            coordTransform->ConvertToLocalCoords(y_loc, tmp, false);
            if (tmp[2] < 0)
                theta[i] = 2. * M_PI - theta[i];
        }
    }

    // for more than one port, we find the minimum angle between the ports

    double theta_ref = 0;
    int tmpPoly = fPortHousing->GetPolyMain();
    double dTheta = 2. * M_PI / fPortHousing->GetPolyMain();

    if (fPortHousing->GetNPorts() > 1) {
        double deltaTheta = 2. * M_PI;
        for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {
            for (unsigned int j = 0; j < i; j++) {
                double tmp = fabs(theta[i] - theta[j]);
                if (tmp < deltaTheta && tmp > 1.e-3) {
                    theta_ref = theta[i];
                    deltaTheta = tmp;
                }
            }
        }

        // now that we have the minimum angle between the ports, we find a theta
        // increment that is amenable to all of the ports (there may be a better
        // way to do this)

        dTheta = 0;
        bool dTheta_isGood = true;

        for (int i = 1; i < 1000; i++) {
            dTheta = deltaTheta / i;
            dTheta_isGood = true;

            // first we check that dTheta divides the unit circle evenly...
            if (fmod(2. * M_PI, dTheta) > 1.e-3 && fabs(fmod(2. * M_PI, dTheta) - dTheta) > 1.e-3) {
                dTheta_isGood = false;
                continue;
            }

            // ...then we check that it divides the angles between all of the ports
            // evenly
            for (unsigned int j = 0; j < fPortHousing->GetNPorts(); j++) {
                if (fmod(theta[j] - theta_ref, dTheta) > 1.e-3 &&
                    fabs(fmod(theta[j] - theta_ref, dTheta) - dTheta) > 1.e-3) {
                    dTheta_isGood = false;
                    continue;
                }
            }

            if (dTheta_isGood)
                break;
        }

        if (!dTheta_isGood) {
            std::stringstream s;
            s << "Unable to find a theta increment that accommodated all of the ports.";
            // KIOManager::GetInstance()->
            //   Message("PortHousing","ComputeEnclosingBoxLengths",s.str(),2);
        }

        // std::cout<<"dTheta: "<<dTheta*180/M_PI<<std::endl;

        // now that we have a minimum theta increment that corresponds to the axes
        // of all of the ports, our main cylinder will be discretized according to
        // this value

        tmpPoly = 1;
        int i = 1;
        while (tmpPoly < fPortHousing->GetPolyMain()) {
            tmpPoly = floor(2. * M_PI / dTheta + .5) * i;
            i++;
        }
        fPortHousing->SetPolyMain(tmpPoly);
    }

    // now that we have properly set the poly parameter for our main cylinder,
    // we compute the enclosing box lengths for the ports

    // ideally, the distance for circular ports is twice the diameter of the
    // aperature.  For rectangular ports, the distance is 1.5 times the length
    // of the aperature.

    std::vector<double> dist_target(fPortHousing->GetNPorts(), 0);

    for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {

        if (const auto* c = dynamic_cast<const KGPortHousing::CircularPort*>(fPortHousing->GetPort(i)))
            dist_target[i] = 4. * c->GetRSub();
        else if (const auto* r = dynamic_cast<const KGPortHousing::RectangularPort*>(fPortHousing->GetPort(i)))
            dist_target[i] = 1.5 * r->GetLength();
    }

    // We now have the ideal sizes for the ports' bounding boxes.  Next, we see
    // if they will fit, and modify them if they don't.

    bool recalculate = true;

    while (recalculate) {
        recalculate = false;

        for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {
            double dist_min = 0;      // an absolute min for the bounding box length
            double dist_max = 1.e30;  // an absolute max for the bounding box length

            if (const auto* c = dynamic_cast<const KGPortHousing::CircularPort*>(fPortHousing->GetPort(i))) {
                // the box must enclose the port, so its side must be at least larger
                // than the diameter of the subordinate cylinder
                dist_min = 2. * c->GetRSub();

                // the box must be smaller than the length of the main cylinder
                // dist_max = sqrt((fAMain[0]-fBMain[0])*(fAMain[0]-fBMain[0]) +
                // 		(fAMain[1]-fBMain[1])*(fAMain[1]-fBMain[1]) +
                // 		(fAMain[2]-fBMain[2])*(fAMain[2]-fBMain[2]));

                // the box cannot run over the edge of the cylinder
                // double dist_upstream = mid[i] + dist_target[i]*.5;

                // if (dist_upstream>x_max)
                //   dist_max = (x_max-mid[i]);

                // double dist_downstream = mid[i] - dist_target[i]*.5;

                // if (dist_downstream<x_min)
                //   dist_max = (mid[i]-x_min);

                // if (fabs(mid[i]-x_min)<dist_max)
                dist_max = 2. * (mid[i] - x_min);

                if (2. * (x_max - mid[i]) < dist_max)
                    dist_max = 2. * (x_max - mid[i]);

                // std::cout<<"Port "<<i<<" dist_upstream, dist_downstream, x_max, x_min: "<<dist_upstream<<" "<<dist_downstream<<" "<<x_max<<" "<<x_min<<std::endl;

                // the box must also be smaller than 1.8 x the radius of the main
                // cylinder
                if (dist_max > 1.8 * fPortHousing->GetRMain())
                    dist_max = 1.8 * fPortHousing->GetRMain();
            }
            else if (const auto* r = dynamic_cast<const KGPortHousing::RectangularPort*>(fPortHousing->GetPort(i))) {
                // the box must enclose the port, so its side must be at least larger
                // than the diameter of the subordinate cylinder
                dist_min = r->GetLength();

                // the box must also be smaller than 1.8 x the radius of the main
                // cylinder
                dist_max = 1.8 * fPortHousing->GetRMain();

                double targetwidth = r->GetWidth() + (dist_target[i] - r->GetLength());

                double dist_upstream = mid[i] + targetwidth * .5;
                if (dist_upstream > x_max)
                    dist_max = (x_max - mid[i]) - r->GetWidth() + r->GetLength();
                double dist_downstream = mid[i] - targetwidth * .5;
                if (dist_downstream < x_min)
                    dist_max = (mid[i] - x_min) - r->GetWidth() + r->GetLength();
            }

            // the box must also be smaller than 1.8 x the radius of the main cylinder
            if (dist_max > 1.8 * fPortHousing->GetRMain())
                dist_max = 1.8 * fPortHousing->GetRMain();

            double dist_act = 0.;

            int nSides = 0;

            // We increment the bounding box length by 2. x the side of the polygon
            // face that comprises the main cylinder.  If it doesn't work, we reduce
            // the target bounding box length and try again.

            while (dist_act < dist_target[i]) {
                if (nSides > tmpPoly / 2) {
                    if (const auto* c = dynamic_cast<const KGPortHousing::CircularPort*>(fPortHousing->GetPort(i))) {
                        dist_target[i] -= c->GetRSub();
                        dist_target[i] *= .9;
                        dist_target[i] += c->GetRSub();
                    }
                    else if (const auto* r =
                                 dynamic_cast<const KGPortHousing::RectangularPort*>(fPortHousing->GetPort(i))) {
                        dist_target[i] -= r->GetLength();
                        dist_target[i] *= .9;
                        dist_target[i] += r->GetLength();
                    }
                    recalculate = true;
                    break;
                }

                nSides += 2;
                dist_act = 2. * fPortHousing->GetRMain() * sin(nSides * (M_PI / tmpPoly));
            }

            // if the calculated bounding box is too small...
            if (dist_act < dist_min) {
                // ...we first check if the port is technically feasible.
                if (2. * fPortHousing->GetRMain() < dist_min) {
                    std::stringstream s;
                    s << "This port cannot exist.";

                    // KIOManager::GetInstance()->
                    //   Message("PortHousing","ComputeEnclosingBoxLengths",s.str(),2);
                }
                // If it is, we try increasing our granularity by increasing the
                // number of sides of the main cylinder polygon
                tmpPoly += (2. * M_PI / dTheta);

                recalculate = true;
                continue;
            }

            // if the calculated bounding box is too large...
            if (dist_act > dist_max) {
                //... we reduce our target length.
                if (const auto* c = dynamic_cast<const KGPortHousing::CircularPort*>(fPortHousing->GetPort(i))) {
                    dist_target[i] -= c->GetRSub();
                    dist_target[i] *= .9;
                    dist_target[i] += c->GetRSub();
                }
                else if (const auto* r =
                             dynamic_cast<const KGPortHousing::RectangularPort*>(fPortHousing->GetPort(i))) {
                    dist_target[i] -= r->GetLength();
                    dist_target[i] *= .9;
                    dist_target[i] += r->GetLength();
                }
                recalculate = true;
                continue;
            }

            // if the calculated bounding box is ok, we set the parameters of the
            // port to this length
            if (const auto* c = dynamic_cast<const KGPortHousing::CircularPort*>(fPortHousing->GetPort(i))) {
                c->SetPolySub(nSides * 4);
                c->SetXDisc((nSides / 3 > 3 ? nSides / 3 : 3));
                c->SetBoxLength(dist_act);
                width[i] = c->GetBoxWidth();
            }
            else if (const auto* r = dynamic_cast<const KGPortHousing::RectangularPort*>(fPortHousing->GetPort(i))) {
                r->SetLengthDisc(nSides);
                r->SetBoxLength(dist_act);
                r->SetBoxWidth(r->GetWidth() + (r->GetBoxLength() - r->GetLength()));
                width[i] = r->GetBoxWidth();
            }
        }

        // if the loop gets to this point and no parameters have been changed, we
        // then check for overlaps between bounding boxes.  Otherwise, we start
        // the box length calculation again.

        if (!recalculate) {
            // we now have box lengths that satisfy the boundary conditions with
            // the main cylinder.  Next, we make sure they don't overlap

            // we start by computing phi
            for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {
                const KGPortHousing::Port* v = fPortHousing->GetPort(i);
                phi[i] = acos(1. - v->GetBoxLength() * v->GetBoxLength() /
                                       (2. * fPortHousing->GetRMain() * fPortHousing->GetRMain()));
            }

            // now, the check

            for (unsigned int i = 0; i < fPortHousing->GetNPorts(); i++) {
                double theta_i_min = theta[i] - phi[i] / 2.;
                double theta_i_max = theta[i] + phi[i] / 2.;
                double x_i_min = mid[i] - width[i] / 2.;
                double x_i_max = mid[i] + width[i] / 2.;

                for (unsigned int j = 0; j < fPortHousing->GetNPorts(); j++) {
                    if (j == i)
                        continue;

                    double theta_j_min = theta[j] - phi[j] / 2.;
                    double theta_j_max = theta[j] + phi[j] / 2.;
                    double x_j_min = mid[j] - width[j] / 2.;
                    double x_j_max = mid[j] + width[j] / 2.;

                    if (ChordsIntersect(theta_i_min, theta_i_max, theta_j_min, theta_j_max) &&
                        LengthsIntersect(x_i_min, x_i_max, x_j_min, x_j_max)) {
                        if (const auto* c =
                                dynamic_cast<const KGPortHousing::CircularPort*>(fPortHousing->GetPort(i))) {
                            dist_target[i] -= c->GetRSub();
                            dist_target[i] *= .9;
                            dist_target[i] += c->GetRSub();
                        }
                        else if (const auto* r =
                                     dynamic_cast<const KGPortHousing::RectangularPort*>(fPortHousing->GetPort(i))) {
                            dist_target[i] -= r->GetLength();
                            dist_target[i] *= .9;
                            dist_target[i] += r->GetLength();
                        }
                        recalculate = true;
                    }
                }
            }
        }
    }

    if (tmpPoly != fPortHousing->GetPolyMain()) {
        std::stringstream s;
        s << "In order to properly match boundaries between the valves of this port, the variable PortHousing::fPortHousing->GetPolyMain() has been modified from "
          << fPortHousing->GetPolyMain() << " to " << tmpPoly << ".";

        // KIOManager::GetInstance()->
        // 	Message("PortHousing","ComputeEnclosingBoxLengths",s.str(),0);
        fPortHousing->SetPolyMain(tmpPoly);
    }
}

//____________________________________________________________________________

void KGPortHousingSurfaceMesher::RectangularPortDiscretizer::DiscretizePort(
    const KGPortHousing::RectangularPort* rectangularPort)
{
    fRectangularPort = rectangularPort;

    double rMain = rectangularPort->GetPortHousing()->GetRMain();

    double merge_length = rectangularPort->GetPortLength() - rMain;

    std::vector<double> xvec(rectangularPort->GetWidthDisc(), 0);
    std::vector<double> yvec(rectangularPort->GetLengthDisc(), 0);
    std::vector<double> zvec(rectangularPort->GetNumDiscSub(), 0);
    std::vector<double> alpha(2. * rectangularPort->GetNumDiscSub(), 0);

    DiscretizeInterval(rectangularPort->GetWidth(), rectangularPort->GetWidthDisc(), 2., xvec);
    DiscretizeInterval(rectangularPort->GetLength(), rectangularPort->GetLengthDisc(), 2., yvec);
    DiscretizeInterval(merge_length, rectangularPort->GetNumDiscSub() - 1, 2., zvec);

    // we compute 2 x the length we are interested in, since the intervals start
    // small, get large, and become small again in a symmetric manner
    DiscretizeInterval(2., 2 * rectangularPort->GetNumDiscSub(), 2., alpha);

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for (int i = 1; i < rectangularPort->GetWidthDisc(); i++)
        xvec[i] += xvec[i - 1];
    for (int i = 1; i < rectangularPort->GetLengthDisc(); i++)
        yvec[i] += yvec[i - 1];
    for (int i = 1; i < rectangularPort->GetNumDiscSub() - 1; i++)
        zvec[i] += zvec[i - 1];
    for (int i = 1; i < rectangularPort->GetNumDiscSub(); i++)
        alpha[i] += alpha[i - 1];

    // we add zero to the front of zvec
    for (int i = rectangularPort->GetNumDiscSub() - 1; i > 0; i--)
        zvec[i] = zvec[i - 1];
    zvec[0] = 0;

    // we flip alpha to go from 1 to zero over <rectangularPort->GetNumDiscSub()> increments
    alpha[0] = 1.;
    for (int i = 1; i < rectangularPort->GetNumDiscSub(); i++)
        alpha[i] = 1. - alpha[i];

    double p0_loc[3];
    double p1_loc[3];
    double p2_loc[3];
    double p3_loc[3];

    double p0[3];
    double p1[3];
    double p2[3];
    double p3[3];

    for (int i = 1; i < rectangularPort->GetNumDiscSub(); i++) {
        p0_loc[1] = p2_loc[1] = -rectangularPort->GetLength() / 2.;

        for (int j = 0; j < rectangularPort->GetLengthDisc(); j++) {
            p1_loc[1] = p3_loc[1] = -rectangularPort->GetLength() / 2. + yvec.at(j);

            // we cast onto the cylinder surface
            p0_loc[2] = (rMain * sqrt(1. - (p0_loc[1] / rMain) * (p0_loc[1] / rMain) * alpha[i - 1]) + zvec.at(i - 1));
            p1_loc[2] = (rMain * sqrt(1. - (p1_loc[1] / rMain) * (p1_loc[1] / rMain) * alpha[i - 1]) + zvec.at(i - 1));
            p2_loc[2] = (rMain * sqrt(1. - (p2_loc[1] / rMain) * (p2_loc[1] / rMain) * alpha[i]) + zvec.at(i));
            p3_loc[2] = (rMain * sqrt(1. - (p3_loc[1] / rMain) * (p3_loc[1] / rMain) * alpha[i]) + zvec.at(i));

            for (int k = 0; k < 2; k++) {
                if (k == 0)
                    p0_loc[0] = p1_loc[0] = p2_loc[0] = p3_loc[0] = -rectangularPort->GetWidth() / 2.;
                else
                    p0_loc[0] = p1_loc[0] = p2_loc[0] = p3_loc[0] = +rectangularPort->GetWidth() / 2.;

                // cast these points into the global frame
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p0_loc, p0, false);
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p1_loc, p1, false);
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p2_loc, p2, false);
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p3_loc, p3, false);

                // now, we cast the global points into triangle-form
                KGMeshTriangle* t = new KGMeshTriangle(p0, p1, p2);
                fPortHousingDiscretizer->AddElement(t);

                t = new KGMeshTriangle(p3, p1, p2);
                fPortHousingDiscretizer->AddElement(t);
            }

            p0_loc[1] = p1_loc[1];
            p2_loc[1] = p3_loc[1];
        }

        p0_loc[0] = p2_loc[0] = -rectangularPort->GetWidth() / 2.;

        for (int j = 0; j < rectangularPort->GetWidthDisc(); j++) {
            p1_loc[0] = p3_loc[0] = -rectangularPort->GetWidth() / 2. + xvec.at(j);

            for (int k = 0; k < 2; k++) {
                if (k == 0)
                    p0_loc[1] = p1_loc[1] = p2_loc[1] = p3_loc[1] = -rectangularPort->GetLength() / 2.;
                else
                    p0_loc[1] = p1_loc[1] = p2_loc[1] = p3_loc[1] = +rectangularPort->GetLength() / 2.;

                // we cast onto the cylinder surface
                p0_loc[2] =
                    (rMain * sqrt(1. - (p0_loc[1] / rMain) * (p0_loc[1] / rMain) * alpha[i - 1]) + zvec.at(i - 1));
                p1_loc[2] =
                    (rMain * sqrt(1. - (p1_loc[1] / rMain) * (p1_loc[1] / rMain) * alpha[i - 1]) + zvec.at(i - 1));
                p2_loc[2] = (rMain * sqrt(1. - (p2_loc[1] / rMain) * (p2_loc[1] / rMain) * alpha[i]) + zvec.at(i));
                p3_loc[2] = (rMain * sqrt(1. - (p3_loc[1] / rMain) * (p3_loc[1] / rMain) * alpha[i]) + zvec.at(i));

                // cast these points into the global frame
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p0_loc, p0, false);
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p1_loc, p1, false);
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p2_loc, p2, false);
                rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p3_loc, p3, false);

                // now, we cast the global points into rectangle-form
                KGMeshRectangle* r = new KGMeshRectangle(p0, p1, p3, p2);
                fPortHousingDiscretizer->AddElement(r);
            }

            p0_loc[0] = p1_loc[0];
            p2_loc[0] = p3_loc[0];
        }
    }

    // Now that the port itself is created, we fill in the bounding box
    // surrounding it

    std::vector<double> xlen(2. * rectangularPort->GetXDisc(), 0);
    std::vector<double> ylen(2. * rectangularPort->GetXDisc(), 0);

    DiscretizeInterval(2. * (rectangularPort->GetBoxWidth() - rectangularPort->GetWidth()),
                       2. * rectangularPort->GetXDisc(),
                       2.,
                       xlen);
    DiscretizeInterval(2. * (rectangularPort->GetBoxLength() - rectangularPort->GetLength()),
                       2. * rectangularPort->GetXDisc(),
                       2.,
                       ylen);

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for (int i = 1; i < rectangularPort->GetXDisc(); i++)
        xlen[i] += xlen[i - 1];
    for (int i = 1; i < rectangularPort->GetXDisc(); i++)
        ylen[i] += ylen[i - 1];

    for (int i = rectangularPort->GetXDisc(); i > 0; i--) {
        xlen[i] = xlen[i - 1] + rectangularPort->GetWidth();
        ylen[i] = ylen[i - 1] + rectangularPort->GetLength();
    }
    xlen[0] = rectangularPort->GetWidth();
    ylen[0] = rectangularPort->GetLength();

    for (int i = 1; i <= rectangularPort->GetXDisc(); i++) {
        for (int j = 0; j < 2. * (rectangularPort->GetLengthDisc() + rectangularPort->GetWidthDisc()); j++) {
            BoundingBoxCoord(j, ylen[i - 1], xlen[i - 1], p0_loc);
            BoundingBoxCoord((j + 1) % (2 * (rectangularPort->GetLengthDisc() + rectangularPort->GetWidthDisc())),
                             ylen[i - 1],
                             xlen[i - 1],
                             p1_loc);
            BoundingBoxCoord(j, ylen[i], xlen[i], p2_loc);
            BoundingBoxCoord((j + 1) % (2 * (rectangularPort->GetLengthDisc() + rectangularPort->GetWidthDisc())),
                             ylen[i],
                             xlen[i],
                             p3_loc);

            // std::cout<<"P0: "<<p0_loc[0]<<" "<<p0_loc[1]<<" "<<p0_loc[2]<<std::endl;

            // cast these points into the global frame
            rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p0_loc, p0, false);
            rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p1_loc, p1, false);
            rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p2_loc, p2, false);
            rectangularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p3_loc, p3, false);

            // now, we cast the global points into triangle-form
            KGMeshTriangle* t = new KGMeshTriangle(p0, p1, p2);
            fPortHousingDiscretizer->AddElement(t);

            t = new KGMeshTriangle(p3, p1, p2);
            fPortHousingDiscretizer->AddElement(t);
        }
    }
}

//______________________________________________________________________________

void KGPortHousingSurfaceMesher::RectangularPortDiscretizer::PowerDistBoxCoord(int i, double length, double width,
                                                                               double* xyz)
{
    // For a given i, i < 2.*fLengthDisc + 2.*fWidthDisc, returns xyz coordinate
    // on a bounding box of dimensions <length>, <width> whose spacing obeys the
    // power law used to discretize the port.

    std::vector<double> xLen(fRectangularPort->GetWidthDisc(), 0);
    std::vector<double> yLen(fRectangularPort->GetLengthDisc(), 0);
    if (i < fRectangularPort->GetLengthDisc()) {
        xyz[0] = width * .5;
        std::vector<double> yLen(fRectangularPort->GetLengthDisc(), 0);
        DiscretizeInterval(length, fRectangularPort->GetLengthDisc(), 2., yLen);
        xyz[1] = -length * .5;
        for (int j = 0; j < i; j++)
            xyz[1] += yLen.at(j);
    }
    else if (i < fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc()) {
        xyz[1] = length * .5;
        std::vector<double> xLen(fRectangularPort->GetWidthDisc(), 0);
        DiscretizeInterval(width, fRectangularPort->GetWidthDisc(), 2., xLen);
        xyz[0] = width * .5;
        for (int j = 0; j < (i - fRectangularPort->GetLengthDisc()); j++)
            xyz[0] -= xLen.at(j);
    }
    else if (i < 2. * fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc()) {
        xyz[0] = -width * .5;
        std::vector<double> yLen(fRectangularPort->GetLengthDisc(), 0);
        DiscretizeInterval(length, fRectangularPort->GetLengthDisc(), 2., yLen);
        xyz[1] = length * .5;
        for (int j = 0; j < (i - (fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc())); j++)
            xyz[1] -= yLen.at(j);
    }
    else {
        xyz[1] = -length * .5;
        std::vector<double> xLen(fRectangularPort->GetWidthDisc(), 0);
        DiscretizeInterval(width, fRectangularPort->GetWidthDisc(), 2., xLen);
        xyz[0] = -width * .5;
        for (int j = 0; j < (i - (2 * fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc())); j++)
            xyz[0] += xLen.at(j);
    }
}

//______________________________________________________________________________

void KGPortHousingSurfaceMesher::RectangularPortDiscretizer::PolygonBoxCoord(int i, double length, double width,
                                                                             double* xyz)
{
    // For a given i, i < 2.*fRectangularPort->GetLengthDisc() + 2.*fRectangularPort->GetWidthDisc(), returns xyz coordinate
    // on a bounding box of dimensions <length>, <width> that conform to the shape
    // of the main cylinder's polygon shape.

    // A lot of this code is borrowed from CircularPort, but we want our indexing
    // to start at a corner of the box.  Therefore, we shift i a bit.
    i = (i + 3 * fRectangularPort->GetLengthDisc() / 2 + 2 * fRectangularPort->GetWidthDisc()) %
        (2 * fRectangularPort->GetLengthDisc() + 2 * fRectangularPort->GetWidthDisc());

    double polyMain = ((double) fRectangularPort->GetPortHousing()->GetPolyMain());

    // len corresponds to the length of a side of the polygon defining the
    // cross-section of the main cylinder
    double len = 2. * fRectangularPort->GetPortHousing()->GetRMain() * tan(M_PI / polyMain);

    len *= length / fRectangularPort->GetBoxLength();

    // phi corresponds to the angle from the vertical of the polygon element
    double phi = M_PI / polyMain;

    // tmp corresponds to the distance along the projected plane
    double tmp = 0.;

    if (i < fRectangularPort->GetLengthDisc() / 2.) {
        if (i != 0)
            for (int j = 1; j <= i; j++)
                tmp += len * cos(phi * (2 * j - 1));

        xyz[0] = width / 2.;
        xyz[1] = tmp;
    }
    else if (i <= fRectangularPort->GetLengthDisc() / 2. + fRectangularPort->GetWidthDisc()) {
        i -= fRectangularPort->GetLengthDisc() / 2.;
        xyz[0] = width / 2. - width / fRectangularPort->GetWidthDisc() * ((double) i);
        xyz[1] = length / 2.;
    }
    else if (i <= fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc()) {
        if (i != fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc())
            for (int j = 1; j <= (fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc() - i); j++)
                tmp += len * cos(phi * (2 * j - 1));
        xyz[0] = -width / 2.;
        xyz[1] = tmp;
    }
    else if (i < 3. / 2. * fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc()) {
        for (int j = 1; j <= (i - (fRectangularPort->GetLengthDisc() + fRectangularPort->GetWidthDisc())); j++)
            tmp += len * cos(phi * (2 * j - 1));
        xyz[0] = -width / 2.;
        xyz[1] = -tmp;
    }
    else if (i <= 2. * fRectangularPort->GetWidthDisc() + 3. / 2. * fRectangularPort->GetLengthDisc()) {
        i -= fRectangularPort->GetWidthDisc() + 3. / 2. * fRectangularPort->GetLengthDisc();
        xyz[0] = -width / 2. + width / fRectangularPort->GetWidthDisc() * ((double) i);
        xyz[1] = -length / 2.;
    }
    else {
        if (i != (2. * fRectangularPort->GetWidthDisc() + 2. * fRectangularPort->GetLengthDisc()))
            for (int j = 1; j <= ((2. * fRectangularPort->GetWidthDisc() + 2. * fRectangularPort->GetLengthDisc()) - i);
                 j++)
                tmp += len * cos(phi * (2 * j - 1));
        xyz[0] = width / 2.;
        xyz[1] = -tmp;
    }
}

//______________________________________________________________________________

void KGPortHousingSurfaceMesher::RectangularPortDiscretizer::BoundingBoxCoord(int i, double length, double width,
                                                                              double* xyz)
{
    // For a given i, i < 2.*fRectangularPort->GetLengthDisc() + 2.*fRectangularPort->GetWidthDisc(), returns xyz coordinate
    // on a bounding box of dimensions <length>, <width>.

    double ratio = (length - fRectangularPort->GetLength()) /
                   (fRectangularPort->GetBoxLength() - fRectangularPort->GetLength());  // 0 at port, 1 at edge

    double xyz_pd[3];
    double xyz_pg[3];

    PowerDistBoxCoord(i, length, width, xyz_pd);
    PolygonBoxCoord(i, length, width, xyz_pg);

    xyz[0] = xyz_pd[0] * (1. - ratio) + xyz_pg[0] * ratio;
    xyz[1] = xyz_pd[1] * (1. - ratio) + xyz_pg[1] * ratio;

    double rMain = fRectangularPort->GetPortHousing()->GetRMain();
    xyz[2] = rMain * sqrt(1. - (xyz[1] / rMain) * (xyz[1] / rMain));
}

//____________________________________________________________________________

void KGPortHousingSurfaceMesher::CircularPortDiscretizer::DiscretizePort(
    const KGPortHousing::CircularPort* circularPort)
{
    // Discretizes the port into triangles and rectangles, and adds them to the
    // mesh.
    fCircularPort = circularPort;

    // we map out points at the intersection of the two cylinders in the local
    // frame ((0,0,0) is at axes intersection, x is along main axis, z is along
    // sub axis)
    std::vector<double> x_int;
    std::vector<double> y_int;
    std::vector<double> z_int;

    double rMain = fCircularPort->GetPortHousing()->GetRMain();

    for (int i = 0; i < circularPort->GetPolySub(); i++) {
        double cosine = cos(2. * M_PI * ((double) i) / circularPort->GetPolySub());
        double sine = sin(2. * M_PI * ((double) i) / circularPort->GetPolySub());
        double sine2 = sine * sine;

        x_int.push_back(circularPort->GetRSub() * cosine);
        y_int.push_back(circularPort->GetRSub() * sine);
        z_int.push_back(sqrt(rMain * rMain - circularPort->GetRSub() * circularPort->GetRSub() * sine2));
    }

    // we then stratify the intersection on the surface of the sub cylinder

    // the length along the subordinate cylinder to discretize in compensation
    // for the asymmetries associated with the intersection
    double merge_length = 0;

    double sub_length = circularPort->GetLength() - rMain;
    if (circularPort->GetRSub() < sub_length)
        merge_length = circularPort->GetRSub();
    else
        merge_length = sub_length;

    // the length of the variation in local z of the intersection
    // double deltaZ = rMain - sqrt(rMain*rMain - circularPort->GetRSub()*circularPort->GetRSub());

    std::vector<double> dz(2 * circularPort->GetXDisc(), 0);
    std::vector<double> alpha(2 * circularPort->GetXDisc(), 0);

    // for each of these interval calculations, we compute 2 x the length we are
    // interested in, since the intervals start small, get large, and become
    // small again in a symmetric manner
    DiscretizeInterval(2 * merge_length, 2 * circularPort->GetXDisc(), 2., dz);
    DiscretizeInterval(2., 2 * circularPort->GetXDisc(), 2., alpha);

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for (int i = 1; i < circularPort->GetXDisc(); i++) {
        dz[i] += dz[i - 1];
        alpha[i] += alpha[i - 1];
    }

    // we flip alpha to go from 1 to zero over <circularPort->GetPolySub()> increments
    alpha[0] = 1.;
    for (int i = 1; i < circularPort->GetXDisc(); i++)
        alpha[i] = 1. - alpha[i];

    std::vector<double> z_low(z_int);
    std::vector<double> z_high(circularPort->GetPolySub(), 0);

    double p0_loc[3];
    double p0[3];

    double p1_loc[3];
    double p1[3];

    double p2_loc[3];
    double p2[3];

    double p3_loc[3];
    double p3[3];

    double a, b, n1[3], n2[3];

    KGMeshTriangle* t;
    KGMeshRectangle* r;

    for (int i = 0; i < circularPort->GetXDisc(); i++) {
        z_high[0] = rMain + dz[i];

        for (int j = 1; j <= circularPort->GetPolySub(); j++) {
            double sine = sin(2. * M_PI * ((double) j) / circularPort->GetPolySub());
            double sine2 = sine * sine;

            if (j < circularPort->GetPolySub())
                z_high[j] =
                    dz[i] +
                    rMain * sqrt(1. - (circularPort->GetRSub() * circularPort->GetRSub() * sine2 / (rMain * rMain)) *
                                          alpha[i]);

            p0_loc[0] = p1_loc[0] = x_int[(j - 1) % circularPort->GetPolySub()];
            p0_loc[1] = p1_loc[1] = y_int[(j - 1) % circularPort->GetPolySub()];
            p0_loc[2] = z_high[(j - 1) % circularPort->GetPolySub()];
            p1_loc[2] = z_low[(j - 1) % circularPort->GetPolySub()];

            p2_loc[0] = p3_loc[0] = x_int[j % circularPort->GetPolySub()];
            p2_loc[1] = p3_loc[1] = y_int[j % circularPort->GetPolySub()];
            p2_loc[2] = z_high[j % circularPort->GetPolySub()];
            p3_loc[2] = z_low[j % circularPort->GetPolySub()];

            // cast these points into the global frame
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p0_loc, p0, false);
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p1_loc, p1, false);
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p2_loc, p2, false);
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p3_loc, p3, false);

            // now, we cast the global points into triangle-form
            t = new KGMeshTriangle(p0, p1, p2);
            fPortHousingDiscretizer->AddElement(t);

            t = new KGMeshTriangle(p3, p1, p2);
            fPortHousingDiscretizer->AddElement(t);
        }

        for (int j = 0; j < circularPort->GetPolySub(); j++)
            z_low[j] = z_high[j];
    }

    // now that the neck of the subordinate cylinder is discretized, we start on
    // the area on the main cylinder surrounding the hole

    DiscretizeInterval(2. * (circularPort->GetBoxLength() / 2. - circularPort->GetRSub()),
                       2 * circularPort->GetXDisc(),
                       2.,
                       dz);

    // by adding each prior interval to the one after it, we get a series of
    // increasing lengths, the last being the sum total of the discretized
    // interval
    for (int i = 1; i < circularPort->GetXDisc(); i++)
        dz[i] += dz[i - 1];

    double r1, r2;

    std::vector<double> theta1(circularPort->GetPolySub(), 0);
    std::vector<double> theta2(circularPort->GetPolySub(), 0);

    // initialize theta1 array
    for (int j = 0; j < circularPort->GetPolySub(); j++)
        theta1[j] = 2. * M_PI * ((double) j) / circularPort->GetPolySub();

    for (int i = 0; i < circularPort->GetXDisc(); i++) {
        if (i == 0)
            r1 = circularPort->GetRSub();
        else
            r1 = circularPort->GetRSub() + dz[i - 1];
        r2 = circularPort->GetRSub() + dz[i];

        for (int j = 1; j <= circularPort->GetPolySub(); j++) {
            // theta will adjust itself according to its distance from the
            // intersection
            if (j < circularPort->GetPolySub())
                theta2[j] = Transition_theta(r2, j);

            // mixed circle-square pattern
            Transition_coord(r2, theta2[(j - 1) % circularPort->GetPolySub()], p0_loc);
            Transition_coord(r1, theta1[(j - 1) % circularPort->GetPolySub()], p1_loc);
            Transition_coord(r2, theta2[j % circularPort->GetPolySub()], p2_loc);
            Transition_coord(r1, theta1[j % circularPort->GetPolySub()], p3_loc);

            // to minimize the lengths of the triangles, we swap coordinates if
            // necessary
            double tmp1 = sqrt((p0_loc[0] - p3_loc[0]) * (p0_loc[0] - p3_loc[0]) +
                               (p0_loc[1] - p3_loc[1]) * (p0_loc[1] - p3_loc[1]) +
                               (p0_loc[2] - p3_loc[2]) * (p0_loc[2] - p3_loc[2]));

            double tmp2 = sqrt((p1_loc[0] - p2_loc[0]) * (p1_loc[0] - p2_loc[0]) +
                               (p1_loc[1] - p2_loc[1]) * (p1_loc[1] - p2_loc[1]) +
                               (p1_loc[2] - p2_loc[2]) * (p1_loc[2] - p2_loc[2]));
            if (tmp2 > tmp1) {
                double tmp3, tmp4;
                for (int k = 0; k < 3; k++) {
                    tmp3 = p0_loc[k];
                    tmp4 = p1_loc[k];
                    p0_loc[k] = p2_loc[k];
                    p1_loc[k] = p3_loc[k];
                    p2_loc[k] = tmp3;
                    p3_loc[k] = tmp4;
                }
            }

            // cast these points into the global frame
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p0_loc, p0, false);
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p1_loc, p1, false);
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p2_loc, p2, false);
            circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p3_loc, p3, false);

            // now, we cast the global points into triangle-form
            t = new KGMeshTriangle(p0, p1, p2);
            fPortHousingDiscretizer->AddElement(t);

            t = new KGMeshTriangle(p1, p2, p3);
            fPortHousingDiscretizer->AddElement(t);
        }
        for (int k = 0; k < circularPort->GetPolySub(); k++)
            theta1[k] = theta2[k];
    }

    // now that the tricky part's over, we have only to finish discretizing the
    // cylindrical portion of the valve

    if (sub_length > merge_length) {
        dz.clear();
        dz.resize(circularPort->GetNumDiscSub());
        DiscretizeInterval(sub_length - merge_length, circularPort->GetNumDiscSub(), 1, dz);

        double theta = 0;
        double theta_last = 0;

        double z = z_high[0];
        double z_last = z_high[0];

        a = circularPort->GetRSub() * sqrt(2. * (1. - cos(2. * M_PI / circularPort->GetPolySub())));

        for (int i = 0; i < circularPort->GetNumDiscSub(); i++) {
            z_last = z;
            z += dz[i];

            b = dz[i];

            for (int j = 1; j <= circularPort->GetPolySub(); j++) {
                theta = Circle_theta(circularPort->GetRSub(), (j % circularPort->GetPolySub()));

                Circle_coord(circularPort->GetRSub(), theta_last, p0_loc);
                p0_loc[2] = z_last;

                Circle_coord(circularPort->GetRSub(), theta, p1_loc);
                p1_loc[2] = z_last;

                Circle_coord(circularPort->GetRSub(), theta_last, p2_loc);
                p2_loc[2] = z;

                // cast these points into the global frame
                circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p0_loc, p0, false);
                circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p1_loc, p1, false);
                circularPort->GetCoordinateTransform()->ConvertToGlobalCoords(p2_loc, p2, false);

                double n1mag = 0;
                double n2mag = 0;

                for (int k = 0; k < 3; k++) {
                    n1[k] = p1[k] - p0[k];
                    n1mag += n1[k] * n1[k];
                    n2[k] = p2[k] - p0[k];
                    n2mag += n2[k] * n2[k];
                }

                n1mag = sqrt(n1mag);
                n2mag = sqrt(n2mag);

                for (int k = 0; k < 3; k++) {
                    n1[k] /= n1mag;
                    n2[k] /= n2mag;
                }

                r = new KGMeshRectangle(a, b, p0, n1, n2);
                fPortHousingDiscretizer->AddElement(r);

                theta_last = theta;
            }
        }
    }
}

//______________________________________________________________________________

double KGPortHousingSurfaceMesher::CircularPortDiscretizer::Circle_theta(double /*r*/, int i)
{
    // This function returns a theta value at regular intervals of a circle.

    return 2. * M_PI * ((double) i) / fCircularPort->GetPolySub();
}

//______________________________________________________________________________

double KGPortHousingSurfaceMesher::CircularPortDiscretizer::Rect_theta(double /*r*/, int i)
{
    // This function returns a theta value that corresponds to regular intervals
    // on the polygon for the main cylinder.

    double increment = (4. * fCircularPort->GetBoxLength()) / fCircularPort->GetPolySub();

    if (i > 7. * fCircularPort->GetPolySub() / 8. || i < fCircularPort->GetPolySub() / 8. ||
        (i > 3. * fCircularPort->GetPolySub() / 8. && i < 5. * fCircularPort->GetPolySub() / 8.)) {
        // we are on the curved side of the rectangle

        double polyMain = ((double) fCircularPort->GetPortHousing()->GetPolyMain());

        // len corresponds to the length of a side of the polygon defining the
        // cross-section of the main cylinder
        double len = 2. * fCircularPort->GetPortHousing()->GetRMain() * tan(M_PI / polyMain);

        // phi corresponds to the angle from the vertical of the polygon element
        double phi = M_PI / polyMain;

        // tmp corresponds to the distance along the projected plane
        double tmp = 0.;

        if (i < fCircularPort->GetPolySub() / 8.) {
            if (i != 0)
                for (int j = 1; j <= i; j++)
                    tmp += len * cos(phi * (2 * j - 1));
            return atan(tmp / (fCircularPort->GetBoxLength() / 2.));
        }
        else if (i > 7. * fCircularPort->GetPolySub() / 8.) {
            if (i != fCircularPort->GetPolySub())
                for (int j = 1; j <= (fCircularPort->GetPolySub() - i); j++)
                    tmp += len * cos(phi * (2 * j - 1));
            return 2. * M_PI - atan(tmp / (fCircularPort->GetBoxLength() / 2.));
        }
        else if (i <= fCircularPort->GetPolySub() / 2) {
            if (i != fCircularPort->GetPolySub() / 2)
                for (int j = 1; j <= (fCircularPort->GetPolySub() / 2 - i); j++)
                    tmp += len * cos(phi * (2 * j - 1));
            return M_PI - atan(tmp / (fCircularPort->GetBoxLength() / 2.));
        }
        else {
            for (int j = 1; j <= (i - fCircularPort->GetPolySub() / 2); j++)
                tmp += len * cos(phi * (2 * j - 1));
            return M_PI + atan(tmp / (fCircularPort->GetBoxLength() / 2.));
        }
    }
    else if (i >= fCircularPort->GetPolySub() / 8. && i <= 3. * fCircularPort->GetPolySub() / 8.) {
        double x = i * increment - fCircularPort->GetBoxLength();
        return M_PI / 2. + atan(x / (fCircularPort->GetBoxLength() / 2.));
    }
    else {
        double x = i * increment - (3. * fCircularPort->GetBoxLength());
        return 3. * M_PI / 2. + atan(x / (fCircularPort->GetBoxLength() / 2.));
    }
}

//______________________________________________________________________________

double KGPortHousingSurfaceMesher::CircularPortDiscretizer::Transition_theta(double r, int i)
{
    // This function varies theta from a circular path to a rectangular one as
    // the point moves away from the intersection.

    double ratio = (r - fCircularPort->GetRSub()) /
                   (fCircularPort->GetBoxLength() / 2. - fCircularPort->GetRSub());  // 0 at the intersection,
    // 1 at the square's edge
    return Circle_theta(r, i) * (1. - ratio) + Rect_theta(r, i) * ratio;
}

//______________________________________________________________________________

void KGPortHousingSurfaceMesher::CircularPortDiscretizer::Circle_coord(double r, double theta, double p[3])
{
    // this function returns a point on a circle with radius <r> at angle <theta>

    p[0] = r * cos(theta);
    p[1] = r * sin(theta);
}

//______________________________________________________________________________

void KGPortHousingSurfaceMesher::CircularPortDiscretizer::Rect_coord(double r, double theta, double p[3])
{
    // this function returns a point on a rectangle inscribing a circle with
    // radius <r> at angle <theta>

    if (theta > 7. * M_PI / 4. || theta < M_PI / 4.) {
        p[0] = r;
        p[1] = r * tan(theta);
    }
    else if (theta >= M_PI / 4. && theta <= 3. * M_PI / 4.) {
        p[0] = r / tan(theta);
        p[1] = r;
    }
    else if (theta > 3. * M_PI / 4. && theta < 5. * M_PI / 4.) {
        p[0] = -r;
        p[1] = -r * tan(theta);
    }
    else {
        p[0] = -r / tan(theta);
        p[1] = -r;
    }
}

//______________________________________________________________________________

void KGPortHousingSurfaceMesher::CircularPortDiscretizer::Transition_coord(double r, double theta, double p[3])
{
    // this function combines Circle_coord and Rect_coord to return a point on the
    // main cylinder that is more circle-like close to the subordinate cylinder,
    // and more square-like farther away.

    // we let the path be a pure circle at the intersection, and a pure square
    // a distance fRSub away from the intersection.  It is therefore assumed that
    // <r> varies from fRSub to fCircularPort->GetBoxLength()/2

    double ratio = (r - fCircularPort->GetRSub()) /
                   (fCircularPort->GetBoxLength() / 2. - fCircularPort->GetRSub());  // 0 at the intersection,
    // 1 at the square's edge

    double p_circ[3];
    double p_rect[3];

    Circle_coord(r, theta, p_circ);
    Rect_coord(r, theta, p_rect);

    p[0] = (1. - ratio) * p_circ[0] + ratio * p_rect[0];
    p[1] = (1. - ratio) * p_circ[1] + ratio * p_rect[1];
    p[2] =
        sqrt(fCircularPort->GetPortHousing()->GetRMain() * fCircularPort->GetPortHousing()->GetRMain() - p[1] * p[1]);
}

//______________________________________________________________________________

bool KGPortHousingSurfaceMesher::ChordsIntersect(double theta1min, double theta1max, double theta2min, double theta2max)
{
    // determines if chord 1 with endpoints <theta1min>, <theta1max> intersects
    // chord 2 with endpoints <theta2min>, <theta2max> on the unit circle.

    // first, normalize the angles
    while (theta1min > 2. * M_PI)
        theta1min -= 2. * M_PI;
    while (theta1min < 0)
        theta1min += 2. * M_PI;
    while (theta1max > 2. * M_PI)
        theta1max -= 2. * M_PI;
    while (theta1max < 0)
        theta1max += 2. * M_PI;
    while (theta2min > 2. * M_PI)
        theta2min -= 2. * M_PI;
    while (theta2min < 0)
        theta2min += 2. * M_PI;
    while (theta2max > 2. * M_PI)
        theta2max -= 2. * M_PI;
    while (theta2max < 0)
        theta2max += 2. * M_PI;

    // if no normalization was needed, this problem's easy:
    if (theta1min < theta1max && theta2min < theta2max)
        return ((theta2min - theta1min > 1.e-5 && theta2min - theta1max < -1.e-5) ||
                (theta2max - theta1min > 1.e-5 && theta2max - theta1max < -1.e-5) ||
                (theta1min - theta2min > 1.e-5 && theta1min - theta2max < -1.e-5) ||
                (theta1max - theta2min > 1.e-5 && theta1max - theta2max < -1.e-5) ||
                ((theta2max + theta2min) * .5 > theta1min && (theta2max + theta2min) * .5 < theta1max));

    // otherwise, we have to shift our frame by performing a cut in a region
    // where we know there is no chord

    bool flip1 = (theta1max < theta1min);
    bool flip2 = (theta2max < theta2min);

    KGLinearCongruentialGenerator pseudoRandomGenerator;

    double random = 0;

    while (true) {
        random = 2. * M_PI * pseudoRandomGenerator.Random();

        if ((flip1 && (random > theta1min || random < theta1max)) ||
            (!flip1 && (random > theta1min && random < theta1max)))
            continue;
        if ((flip2 && (random > theta2min || random < theta2max)) ||
            (!flip2 && (random > theta2min && random < theta2max)))
            continue;
        break;
    }

    // so <random> is a point on the line from 0 to 2 Pi that does not intersect
    // either chord.  Let's shift it to zero

    theta1min -= random;
    if (theta1min < 0)
        theta1min += 2. * M_PI;
    theta1max -= random;
    if (theta1max < 0)
        theta1max += 2. * M_PI;
    theta2min -= random;
    if (theta2min < 0)
        theta2min += 2. * M_PI;
    theta2max -= random;
    if (theta2max < 0)
        theta2max += 2. * M_PI;

    return ((theta2min - theta1min > 1.e-5 && theta2min - theta1max < -1.e-5) ||
            (theta2max - theta1min > 1.e-5 && theta2max - theta1max < -1.e-5) ||
            (theta1min - theta2min > 1.e-5 && theta1min - theta2max < -1.e-5) ||
            (theta1max - theta2min > 1.e-5 && theta1max - theta2max < -1.e-5) ||
            ((theta2max + theta2min) * .5 > theta1min && (theta2max + theta2min) * .5 < theta1max));
}

//______________________________________________________________________________

bool KGPortHousingSurfaceMesher::LengthsIntersect(double x1min, double x1max, double x2min, double x2max)
{
    // determines if chord 1 with endpoints <theta1min>, <theta1max> intersects
    // chord 2 with endpoints <theta2min>, <theta2max> on the unit circle.

    return ((x2min - x1min > 1.e-5 && x2min - x1max < -1.e-5) || (x2max - x1min > 1.e-5 && x2max - x1max < -1.e-5) ||
            (x1min - x2min > 1.e-5 && x1min - x2max < -1.e-5) || (x1max - x2min > 1.e-5 && x1max - x2max < -1.e-5) ||
            ((x2max + x2min) * .5 > x1min && (x2max + x2min) * .5 < x1max));
}

}  // namespace KGeoBag
