//
// Created by trost on 03.06.15.
//

#ifndef KASPER_QUADGAUSSLEGENDRE_H
#define KASPER_QUADGAUSSLEGENDRE_H


namespace Kassiopeia
{

class QuadGaussLegendre
{

  public:
    QuadGaussLegendre(){};

    virtual ~QuadGaussLegendre(){};

    /* \brief Numerical integration routine:
         * quad=integral of function f from a to b, with Gauss.Legendre method.
         * n: number of integration nodes.
        */
    template<class XFunctorType> static double Integrate32(XFunctorType& f, double a, double b, int n)
    {
        int j;
        double A, B, sum;
        static double x2[1] = {0.577350269189626};
        static double w2[1] = {1.};
        static double x3[2] = {0., 0.774596669241483};
        static double w3[2] = {0.888888888888889, 0.555555555555556};
        static double x4[2] = {0.339981043584856, 0.861136311594053};
        static double w4[2] = {0.652145154862546, 0.347854845137454};
        static double x6[3] = {0.238619186083197, 0.661209386466265, 0.932469514203152};
        static double w6[3] = {0.467913934572691, 0.360761573048139, 0.171324492379170};
        static double x8[4] = {0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536};
        static double w8[4] = {0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376};
        static double x16[8] = {0.09501250983763744,
                                0.28160355077925891,
                                0.45801677765722739,
                                0.61787624440264375,
                                0.75540440835500303,
                                0.86563120238783174,
                                0.94457502307323258,
                                0.98940093499164993};
        static double w16[8] = {0.189450610455068496,
                                0.182603415044923589,
                                0.169156519395002532,
                                0.149595988816576731,
                                0.124628971255533872,
                                0.095158511682492785,
                                0.062253523938647892,
                                0.027152459411754095};
        static double x32[16] = {0.048307665687738316,
                                 0.144471961582796493,
                                 0.239287362252137075,
                                 0.331868602282127650,
                                 0.421351276130635345,
                                 0.506899908932229390,
                                 0.587715757240762329,
                                 0.663044266930215201,
                                 0.732182118740289680,
                                 0.794483795967942407,
                                 0.849367613732569970,
                                 0.896321155766052124,
                                 0.934906075937739689,
                                 0.964762255587506431,
                                 0.985611511545268335,
                                 0.997263861849481564};
        static double w32[16] = {0.09654008851472780056,
                                 0.09563872007927485942,
                                 0.09384439908080456564,
                                 0.09117387869576388471,
                                 0.08765209300440381114,
                                 0.08331192422694675522,
                                 0.07819389578707030647,
                                 0.07234579410884850625,
                                 0.06582222277636184684,
                                 0.05868409347853554714,
                                 0.05099805926237617619,
                                 0.04283589802222680057,
                                 0.03427386291302143313,
                                 0.02539206530926205956,
                                 0.01627439473090567065,
                                 0.00701861000947009660};
        if (n <= 2)
            n = 2;
        else if (n >= 5 && n <= 6)
            n = 6;
        else if (n >= 7 && n <= 8)
            n = 8;
        else if (n >= 9 && n <= 16)
            n = 16;
        else if (n > 16)
            n = 32;
        A = (b - a) / 2.;
        B = (b + a) / 2.;
        sum = 0.;
        //printf("n=%12i \t\n",n);
        if (n == 2) {
            sum = w2[0] * f(B + A * x2[0]) + w2[0] * f(B - A * x2[0]);
        }
        else if (n == 3) {
            sum = w3[0] * f(B + A * x3[0]) + w3[1] * f(B + A * x3[1]) + w3[1] * f(B - A * x3[1]);
        }
        else if (n == 4) {
            for (j = 0; j <= 1; j++)
                sum += w4[j] * f(B + A * x4[j]);
            for (j = 0; j <= 1; j++)
                sum += w4[j] * f(B - A * x4[j]);
        }
        else if (n == 6) {
            for (j = 0; j <= 2; j++)
                sum += w6[j] * f(B + A * x6[j]);
            for (j = 0; j <= 2; j++)
                sum += w6[j] * f(B - A * x6[j]);
        }
        else if (n == 8) {
            for (j = 0; j <= 3; j++)
                sum += w8[j] * f(B + A * x8[j]);
            for (j = 0; j <= 3; j++)
                sum += w8[j] * f(B - A * x8[j]);
        }
        else if (n == 16) {
            for (j = 0; j <= 7; j++)
                sum += w16[j] * f(B + A * x16[j]);
            for (j = 0; j <= 7; j++)
                sum += w16[j] * f(B - A * x16[j]);
        }
        else if (n == 32) {
            for (j = 0; j <= 15; j++)
                sum += w32[j] * f(B + A * x32[j]);
            for (j = 0; j <= 15; j++)
                sum += w32[j] * f(B - A * x32[j]);
        }
        return sum * A;
    };

    template<class XFunctorType> static double Integrate(XFunctorType& f, double a, double b, int n)
    {
        double Integral, xmin, xmax, del;
        if (n <= 32)
            Integral = Integrate32(f, a, b, n);
        else {
            int imax = n / 32 + 1;
            del = (b - a) / imax;

            Integral = 0.;
            for (int i = 1; i <= imax; i++) {
                xmin = a + del * (i - 1);
                xmax = xmin + del;
                Integral += Integrate32(f, xmin, xmax, 32);
            }
        }
        return Integral;
    };

    /** \brief Integral of function f from 0 to infinity
         *  Integral= sum_i ( integral from a_i to b_i)
         * step1=first integral step   (0 to step1)
         * tol: tolerance limit; integration procedure stops if abs(Del/Integral)<tolerance;
         * tol should be much smaller than 1 (positive)
         * N: number of integration points for one integration step (e.g.:  8, 16,  32)
         * xlimit: the integration procedure does not stop until the integration limit b_i is  not larger than xlimit
         */
    template<class XFunctorType>
    static double IntegrateH(XFunctorType& f, double step1, double xlimit, double tol, int N)
    {
        double a, b, Integral, Del, err;
        int i;

        a = 0.;
        b = step1;
        Integral = 0.;
        i = 1;
        err = 1.;
        do {
            Del = Integrate(f, a, b, N);
            Integral += Del;
            i += 1;
            a = b;
            b = step1 * i * i;
            if (i > 5)
                err = fabs(Del / Integral);
        } while (b < xlimit || err > tol);

        return Integral;
    };
};

}  // namespace Kassiopeia

#endif  //KASPER_QUADGAUSSLEGENDRE_H
