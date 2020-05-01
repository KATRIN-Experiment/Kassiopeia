#include "KElectrostaticAnalyticRectangleIntegrator.hh"

namespace KEMField
{
/**
 * \image html potentialFromRectangle.gif
 * Returns the electric potential at a point P (P[0],P[1],P[2]) due to the
 * collection of rectangles by computing the following integral for each copy:
 * \f{eqnarray*}{
 * V(\vec{P}) &= \left( \frac{\sigma}{4 \pi \epsilon_0} \right) & \int_{-u_p}^{-u_p+a} \int_{-v_p}^{-v_p+b} \frac{1}{r}\cdot dy \cdot dx = \\
 * &= \left( \frac{\sigma}{4 \pi \epsilon_0} \right) & \left( F\left( (-u_p+a),(0v_p+b),w_p \right) - F\left( -u_p,(-v_p+b),w_b \right) \right. \\
 * && \left. -F\left( (-u_p+a),-v_p,w_p \right) + F\left( -u_p,-v_p,w_p \right) \right)
 * \f}
 * where
 * \f{eqnarray*}{
 * F(x,y,z) = \int \ln(y+r) \cdot dx &=& z \cdot \arctan\left(\frac{x}{z}\right) - z \cdot \arctan\left(\frac{xy}{zr}\right) - x +\\
 * && + y \ln(x+r)+x \ln(y+r).
 * \f}
 * and
 * \f[
 * r = \sqrt{(u-u_p)^2+(v-v_p)^2+w_p^2} = \sqrt{x^2+y^2+w_p^2}
 * \f]
 * and the coordinates are as described in the above image.
 */
double KElectrostaticAnalyticRectangleIntegrator::Potential(const KRectangle* source, const KPosition& P) const
{
    KThreeVector p = P - source->GetP0();
    double uP = p.Dot(source->GetN1());
    double vP = p.Dot(source->GetN2());
    double w = p.Dot(source->GetN3());

    double xmin, xmax, ymin, ymax;  // integration parameters
    xmin = -uP;
    xmax = -uP + source->GetA();
    ymin = -vP;
    ymax = -vP + source->GetB();

    double I = (Integral_ln(xmax, ymax, w) - Integral_ln(xmin, ymax, w) - Integral_ln(xmax, ymin, w) +
                Integral_ln(xmin, ymin, w));

    return I / (4. * M_PI * KEMConstants::Eps0);
}

KThreeVector KElectrostaticAnalyticRectangleIntegrator::ElectricField(const KRectangle* source,
                                                                      const KPosition& P) const
{
    KThreeVector p = P - source->GetP0();
    double uP = p.Dot(source->GetN1());
    double vP = p.Dot(source->GetN2());
    double w = p.Dot(source->GetN3());

    double xmin, xmax, ymin, ymax;  // integration parameters
    xmin = -uP;
    xmax = -uP + source->GetA();
    ymin = -vP;
    ymax = -vP + source->GetB();

    double prefac = 1. / (4. * KEMConstants::Pi * KEMConstants::Eps0);

    KThreeVector field_local(0., 0., 0.);

    field_local[0] = prefac * EFieldLocalXY(xmin, xmax, ymin, ymax, w);
    field_local[1] = prefac * EFieldLocalXY(ymin, ymax, xmin, xmax, w);

    double tmin = w / ymin;
    double tmax = w / ymax;
    double sign_z = 1.;
    if (w < 0)
        sign_z = -1.;
    if (fabs(w) < 1.e-13) {
        if (xmin < 0. && xmax > 0. && ymin < 0. && ymax > 0.) {
            //changed 12/8/14 JB
            //We are probably on the surface of the element
            //so ignore any z displacement (which may be a round off error)
            //and always pick a consistent direction (aligned with normal vector)
            field_local[2] = 1.0 / (2. * KEMConstants::Eps0);
            //field_local[2] = sign_z/(2.*KEMConstants::Eps0);
        }
        else {
            field_local[2] = 0.;
        }
    }
    else {
        if (((tmin > 0) - (tmin < 0)) != ((tmax > 0) - (tmax < 0)))
            field_local[2] =
                prefac * sign_z *
                fabs(EFieldLocalZ(xmin, xmax, 0, fabs(ymin), w) + EFieldLocalZ(xmin, xmax, 0, fabs(ymax), w));
        else
            field_local[2] = prefac * sign_z * fabs(EFieldLocalZ(xmin, xmax, ymax, ymin, w));
    }

    KThreeVector field(0., 0., 0.);

    for (unsigned int i = 0; i < 3; i++)
        field[i] = (source->GetN1()[i] * field_local[0] + source->GetN2()[i] * field_local[1] +
                    source->GetN3()[i] * field_local[2]);
    return field;
}

/**
 * Computes the following indefinite integral analytically:
 *
 * \f[
 * \int_t dx \ln(y+r), r = \sqrt{x^{2}+y^{2}+w^{2}}.
 * \f]
 */
double KElectrostaticAnalyticRectangleIntegrator ::Integral_ln(double x, double y, double w) const
{
    double r, r0, ret, xa, c1, c2, c3;
    r = sqrt(fabs(x * x + y * y + w * w));
    r0 = sqrt(fabs(y * y + w * w));
    xa = fabs(x);
    if (xa < 1.e-10)
        c1 = 0.;
    else
        c1 = xa * log(fabs(y + r) + 1.e-12);
    if (fabs(y) < 1.e-12)
        c2 = 0.;
    else
        c2 = y * log(fabs((xa + r) / r0) + 1.e-12);
    if (fabs(w) < 1.e-12)
        c3 = 0.;
    else
        c3 = w * (atan(xa / w) + atan(y * w / (x * x + w * w + xa * r)) - atan(y / w));
    ret = c1 + c2 - xa + c3;
    if (x < 0.)
        ret = -ret;
    return ret;
}

double KElectrostaticAnalyticRectangleIntegrator::EFieldLocalXY(double x1, double x2, double y1, double y2,
                                                                double z) const
{
    // Computes the x (or y) component of the electric field in local coordinates
    // (where the rectangle lies in the x-y plane, and the field point lies on
    // the z-axis).

    double a1 = y2 + sqrt(y2 * y2 + x1 * x1 + z * z);
    double a2 = y2 + sqrt(y2 * y2 + x2 * x2 + z * z);
    double a3 = y1 + sqrt(y1 * y1 + x1 * x1 + z * z);
    double a4 = y1 + sqrt(y1 * y1 + x2 * x2 + z * z);

    if (fabs(z) < 1.e-14) {
        if (fabs(x1) < 1.e-14) {
            a1 = fabs(y1);
            a3 = fabs(y2);
        }
        if (fabs(x2) < 1.e-14) {
            a2 = fabs(y1);
            a4 = fabs(y2);
        }
    }

    if (fabs(a1 - a3) < 1.e-14)
        a1 = a3 = 1.;

    if (fabs(a2 - a4) < 1.e-14)
        a2 = a4 = 1.;

    return log((a2 * a3) / (a1 * a4));
}

double KElectrostaticAnalyticRectangleIntegrator::EFieldLocalZ(double x1, double x2, double y1, double y2,
                                                               double z) const
{
    // Computes the z component of the electric field in local coordinates (where
    // the rectangle lies in the x-y plane, and the field point lies on the
    // z-axis).

    double t1 = (fabs(y1) > 1.e-15 ? z / y1 : 1.e15);
    double t2 = (fabs(y2) > 1.e-15 ? z / y2 : 1.e15);

    double g1 = sqrt(((x2 * x2 + z * z) * t1 * t1 + z * z) / (x2 * x2));
    double g2 = sqrt(((x1 * x1 + z * z) * t1 * t1 + z * z) / (x1 * x1));
    double g3 = sqrt(((x2 * x2 + z * z) * t2 * t2 + z * z) / (x2 * x2));
    double g4 = sqrt(((x1 * x1 + z * z) * t2 * t2 + z * z) / (x1 * x1));

    if (x2 < 0.)
        g1 = -g1;
    if (x1 < 0.)
        g2 = -g2;
    if (x2 < 0.)
        g3 = -g3;
    if (x1 < 0.)
        g4 = -g4;

    return atan(g3) - atan(g4) - atan(g1) + atan(g2);
}

}  // namespace KEMField
