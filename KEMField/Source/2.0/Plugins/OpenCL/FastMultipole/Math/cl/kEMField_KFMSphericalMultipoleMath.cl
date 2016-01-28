#ifndef KFMSphericalMultipole_Defined_H
#define KFMSphericalMultipole_Defined_H


#include "kEMField_defines.h"

#define SQRT_OF_THREE 1.7320508075688772935274463


//______________________________________________________________________________

//copy the data in the vector in, to out
void
Copy(int size, CL_TYPE* in, CL_TYPE* out)
{
    for(int i=0; i<size; i++)
    {
        out[i] = in[i];
    }
}

//copy the data in the vector in, to out
void
Copy2(int size, CL_TYPE2* in, CL_TYPE2* out)
{
    for(int i=0; i<size; i++)
    {
        out[i] = in[i];
    }
}



//______________________________________________________________________________
//simple cartesian to spherical conversion utilities
CL_TYPE
Radius(CL_TYPE4 O, CL_TYPE4 P)
{
    return sqrt((P.s0-O.s0)*(P.s0-O.s0)+(P.s1-O.s1)*(P.s1-O.s1)+(P.s2-O.s2)*(P.s2-O.s2));
}


CL_TYPE
Phi(CL_TYPE4 O, CL_TYPE4 P)
{
    return atan2((P.s1-O.s1),(P.s0-O.s0));
};

CL_TYPE
CosTheta(CL_TYPE4 O, CL_TYPE4 P)
{
    CL_TYPE r = Radius(O,P);
    return ((P.s2-O.s2)/r);
};


CL_TYPE
RadiusSingle(CL_TYPE4 P)
{
    return sqrt((P.s0)*(P.s0)+(P.s1)*(P.s1)+(P.s2)*(P.s2));
}


CL_TYPE
PhiSingle(CL_TYPE4 P)
{
    return atan2((P.s1),(P.s0));
};

CL_TYPE
CosThetaSingle(CL_TYPE4 P)
{
    CL_TYPE r = RadiusSingle(P);
    return (P.s2/r);
};


//______________________________________________________________________________

//base case P_m^m(x), this should not be called directly as it is unnormalized
//use ALP_nm instead
CL_TYPE
ALP_mm(int m, CL_TYPE x)
{
    if(m == 0)
    {
        return 1.0;
    }

    CL_TYPE u = sqrt( (1.0-x)*(1.0+x) );
    CL_TYPE p_mm = SQRT_OF_THREE*u;

    for(int i=2; i <= m; i++)
    {
        p_mm *= u*sqrt((2.0*i + 1.0)/(2.0*i));
    }

    return p_mm;

}


//______________________________________________________________________________

//returns the Schmidt semi-normalized spherical harmonics
CL_TYPE
ALP_nm(int n, int m, CL_TYPE x)
{
    CL_TYPE norm;

    if(m == 0)
    {
        norm = rsqrt(2.0*n + 1.0);
    }
    else
    {
        norm = rsqrt( 2.0*(2.0*n + 1.0) );
        if( (m & 1) != 0)
        {
            norm *= -1.0; //condon-shortley phase convention
        }
    }

    if(n == m)
    {
        return norm*ALP_mm(m, x);
    }
    else if(n == m + 1)
    {
        return norm*x*sqrt(2.0*m + 3.0)*ALP_mm(m,x);
    }
    else
    {
        CL_TYPE p_b = ALP_mm(m,x);
        CL_TYPE p_a =  x*sqrt(2.0*m + 3.0)*p_b;
        CL_TYPE plm, alm, blm, dm, dl;
        dm = m;

        for(int l = m+2; l <= n; l++)
        {
            dl = l;
            alm = sqrt(( (2.0*dl - 1.0)*(2.0*dl + 1.0) )/( (dl - dm)*(dl + dm) ));
            blm = sqrt( ((2.0*dl + 1.0)*(dl + dm - 1.0)*(dl - dm - 1.0))/((dl - dm)*(dl + dm)*(2.0*dl - 3.0)) );
            plm = alm*x*p_a - blm*p_b;
            p_b = p_a;
            p_a = plm;
        }

        return norm*plm;
    }
}



//______________________________________________________________________________
//computes array of associated legendre polynomials (schmidt semi-normalized)

void
ALP_nm_array(int n_max, CL_TYPE x,  CL_TYPE* val)
{
    val[0] = 1.0;
    if(n_max == 0){return;};

    CL_TYPE u = sqrt( (1.0-x)*(1.0+x) );
    val[1] = x*SQRT_OF_THREE;
    val[2] = u*SQRT_OF_THREE;
    if(n_max == 1){return;};

    //evaluate the base cases for each p(m,m) and p(m, m-1) up to m = n_max
    int l, m, si, si_a, si_b;
    CL_TYPE p_mm = val[2];
    CL_TYPE alm, blm, dm, dl;
    for(m=2; m <= n_max; m++)
    {
        dm = m;
        //first base case value P(m,m)
        si_a = (m*(m+1))/2 + m;
        p_mm *= u*sqrt((2.0*dm + 1.0)/(2.0*dm));
        val[si_a] = p_mm;

        //second base case value P(m, m-1)
        si_b = ((m)*(m+1))/2 + m - 1;
        si_a = (m*(m-1))/2 + m - 1;
        val[si_b] = x*sqrt(2.0*(dm-1.0) + 3.0)*val[si_a];
    }

    //do reccurance over the rest of the whole array for (l,m) > (1,1)
    for(m=0; m <= n_max; m++)
    {
        dm = m;
        for(l = m+2; l <= n_max; l++)
        {
            dl = l;
            si = (l*(l+1))/2 + m;
            si_a = (l*(l-1))/2 + m;
            si_b = ((l-2)*(l-1))/2 + m;

            alm = sqrt(( (2.0*dl - 1.0)*(2.0*dl + 1.0) )/( (dl - dm)*(dl + dm) ));
            blm = sqrt( ((2.0*dl + 1.0)*(dl + dm - 1.0)*(dl - dm - 1.0))/((dl - dm)*(dl + dm)*(2.0*dl - 3.0)) );

            val[si] = alm*x*val[si_a] - blm*val[si_b];
        }
    }

    //normalize
    CL_TYPE norm;
    for(l=0; l<= n_max; l++)
    {
        dl = l;
        for(m=0; m <= l; m++)
        {
            if(m == 0)
            {
                norm = rsqrt(2.0*dl + 1.0);
            }
            else
            {
                norm = rsqrt( 2.0*(2.0*dl + 1.0) );
                if( (m & 1) != 0)
                {
                    norm *= -1.0;
                }
            }

            si = (l*(l+1))/2 + m;
            val[si] *= norm;
        }
    }

}


//______________________________________________________________________________


void
ALP_nm_unormalized_array(int n_max, CL_TYPE x,  CL_TYPE* val)
{
    val[0] = 1.0;
    if(n_max == 0){return;};

    CL_TYPE u = sqrt( (1.0-x)*(1.0+x) );
    val[2] = u*SQRT_OF_THREE;
    val[1] = x*SQRT_OF_THREE;
    if(n_max == 1){return;};

    //evaluate the base cases for each p(m,m) and p(m, m-1) up to m = n_max
    int l, m, si, si_a, si_b;
    CL_TYPE p_mm = val[2];
    CL_TYPE alm, blm, dl, dm;
    for(m=2; m <= n_max; m++)
    {
        dm = m;
        //first base case value P(m,m)
        si_a = (m*(m+1))/2 + m;
        p_mm *= u*sqrt((2.0*dm + 1.0)/(2.0*dm));
        val[si_a] = p_mm;

        //second base case value P(m, m-1)
        si_b = ((m)*(m+1))/2 + m - 1;
        si_a = (m*(m-1))/2 + m - 1;
        val[si_b] = x*sqrt(2.0*(dm-1.0) + 3.0)*val[si_a];
    }

    //do reccurance over the rest of the whole array for (l,m) > (1,1)
    for(m=0; m <= n_max; m++)
    {
        dm = m;
        for(l = m+2; l <= n_max; l++)
        {
            dl = l;
            si = (l*(l+1))/2 + m;
            si_a = (l*(l-1))/2 + m;
            si_b = ((l-2)*(l-1))/2 + m;

            alm = sqrt(( (2.0*dl - 1.0)*(2.0*dl + 1.0) )/ ( (dl - dm)*(dl + dm) ));
            blm = sqrt( ((2.0*dl + 1.0)*(dl + dm - 1.0)*(dl - dm - 1.0))/ ((dl - dm)*(dl + dm)*(2.0*dl - 3.0)) );

            val[si] = alm*x*val[si_a] - blm*val[si_b];
        }
    }

}

//______________________________________________________________________________

//returns the un-normalized associated legendre polynomials
CL_TYPE
ALP_nm_unormalized(int n, int m, CL_TYPE x)
{
    CL_TYPE dl = n;
    CL_TYPE dm = m;

    if( (n < 0) || (n< m) || (fabs(x) > 1.0) )
    {
        return NAN;
    }

    if(n == m)
    {
        return ALP_mm(m, x);
    }
    else if(n == m + 1)
    {
        return x*sqrt(2.0*dm + 3.0)*ALP_mm(m,x);
    }
    else
    {
        CL_TYPE p_b = ALP_mm(m,x);
        CL_TYPE p_a =  x*sqrt(2.0*dm + 3)*p_b;
        CL_TYPE plm, alm, blm;

        for(int l = m+2; l <= n; l++)
        {
            dl = l;
            alm = sqrt(( (2.0*dl - 1.0)*(2.0*dl + 1.0) )/ ( (dl - dm)*(dl + dm) ));
            blm = sqrt( ((2.0*dl + 1.0)*(dl + dm - 1.0)*(dl - dm - 1.0))/ ((dl - dm)*(dl + dm)*(2.0*dl - 3.0)) );
            plm = alm*x*p_a - blm*p_b;
            p_b = p_a;
            p_a = plm;
        }

        return plm;
    }
}


//______________________________________________________________________________


//returns the derivative of the Schmidt semi-normalized associated legendre polynomials
CL_TYPE ALPDerv_nm(int n, int m, CL_TYPE x)
{
    CL_TYPE dn, dm;
    dn = n;
    dm = m;

    if( (n < 0) || (n < m) || (fabs(x) > 1.0) )
    {
        return NAN;
    }

    if( n == 0 )
    {
        return 0.0;
    }
    else
    {
        if( n == m)
        {
            CL_TYPE inv_u = 1.0/sqrt( (1.0 - x)*(1.0 + x) );
            CL_TYPE norm;

            if(m == 0)
            {
                norm = rsqrt(2.0*dn + 1.0);
            }
            else
            {
                norm = rsqrt( 2.0*(2.0*dn + 1.0) );
                if(m%2 != 0)
                {
                    norm *= -1.0; //condon-shortley phase
                }
            }
            return dm*norm*x*inv_u*ALP_mm(m,x);

        }
        else
        {

            CL_TYPE norm;

            if(m == 0)
            {
                norm = rsqrt(2.0*dn + 1.0);
            }
            else
            {
                norm = rsqrt( 2.0*(2.0*dn + 1.0) );
                if(m%2 != 0)
                {
                    norm *= -1.0;  //condon-shortley phase
                }
            }

            CL_TYPE inv_u = 1.0/sqrt( (1.0 - x)*(1.0 + x) );
            CL_TYPE fnm = sqrt( (dn*dn - dm*dm)*(2.0*dn + 1.0) / (2.0*dn - 1.0) );
            CL_TYPE num = dn*x*ALP_nm_unormalized(n,m,x) - fnm*ALP_nm_unormalized(n-1, m, x);
            return norm*num*inv_u;
        }
    }
}

//______________________________________________________________________________

void ALPAndFirstDerv_array(int n_max, CL_TYPE x,  CL_TYPE* PlmVal,  CL_TYPE* PlmDervVal)
{
    ALP_nm_unormalized_array(n_max, x, PlmVal);

    CL_TYPE u = sqrt( (1.0 - x)*(1.0 + x) );
    CL_TYPE inv_u;
    CL_TYPE dm, dl;

    inv_u = 1.0/u;

    CL_TYPE flm, plm, plm_lm1;
    int si_a, si_b;

    for(int m = 0; m <= n_max; m++)
    {
        dm = m;
        for(int l = n_max; l >= m+1; l-- )
        {
            dl = l;
            si_a = (l*(l+1))/2 + m;
            plm = PlmVal[si_a];
            si_b = (l*(l-1))/2 + m;
            plm_lm1 = PlmVal[si_b];
            flm = sqrt( ((dl*dl - dm*dm)*(2.0*dl + 1.0)) / (2.0*dl - 1.0) );
            PlmDervVal[si_a] = inv_u*(dl*x*plm - flm*plm_lm1);
        }
    }


    //take care of sectoral derivatives last
    for(int m = 0; m <= n_max; m++)
    {
        dm = m;
        int si_a = m*(m+1)/2 + m;
        PlmDervVal[si_a] = dm*x*inv_u*PlmVal[si_a];
    }

    //normalize
    CL_TYPE norm;
    for(int l=0; l<= n_max; l++)
    {
        dl = l;
        for(int m=0; m <= l; m++)
        {
            dm = m;
            if(m == 0)
            {
                norm = rsqrt(2.0*dl + 1.0);
            }
            else
            {
                norm = rsqrt( 2.0*(2.0*dl + 1.0) );
                if(m%2 != 0)
                {
                    norm *= -1.0; //condon-shortley phase
                }
            }

            si_a = (l*(l+1))/2 + m;
            PlmVal[si_a] *= norm;
            PlmDervVal[si_a] *= norm;
        }
    }


}


//______________________________________________________________________________
//computes the integral of sec(x)^{n}, n >= 2


CL_TYPE
I_secn(int n, CL_TYPE lower_limit, CL_TYPE upper_limit)
{
    CL_TYPE tan_up = tan(upper_limit);
    CL_TYPE tan_low = tan(lower_limit);
    CL_TYPE sec_up = (1.0/cos(upper_limit));
    CL_TYPE sec_low = (1.0/cos(lower_limit));
    CL_TYPE sin_diff = sin(upper_limit - lower_limit);

    if(n == 2)
    {
        return sec_up*sec_low*sin_diff;
    }



    CL_TYPE a, b, up, low, result;

    if( (n & 1) == 0) // only even powers involved
    {
        CL_TYPE s_2 = sec_up*sec_low*sin_diff;
        b = s_2;
        for(int i=2; i < n; )
        {
            up = tan_up*(pow(sec_up, (CL_TYPE)(i ) ) )*(1.0/( (CL_TYPE)(i) + 1.0));
            low = tan_low*(pow(sec_low, (CL_TYPE)(i) ) )*(1.0/( (CL_TYPE)(i) + 1.0));
            a = up - low;
            result = a + (((CL_TYPE)(i) )/((CL_TYPE)(i) + 1.0))*b;
            b = result;
            i += 2;
        }
    }
    else //only odd powers involved
    {
        CL_TYPE s_1 = log( fabs( tan_up + sec_up )/fabs( tan_low + sec_low ) ) ;//up - low;
        b = s_1;
        for(int i=1; i < n; )
        {
            up = tan_up*(pow(sec_up, (CL_TYPE)(i )   ) )*(1.0/( (CL_TYPE)(i) + 1.0));
            low = tan_low*(pow(sec_low, (CL_TYPE)(i) ) )*(1.0/( (CL_TYPE)(i) + 1.0));
            a = up - low;
            result = a + (((CL_TYPE)(i) )/((CL_TYPE)(i) + 1.0))*b;
            b = result;
            i += 2;
        }
    }

    return result;
}


//______________________________________________________________________________
//computes the integral of sin(x)/cos(x)^{n} with n > 1

CL_TYPE
I_trig1(int n, CL_TYPE lower_limit, CL_TYPE upper_limit)
{
    if( n <= 0 )
    {
        return 0;
    }

    if (n == 1)
    {
        return log( cos(lower_limit)/cos(upper_limit) );
    }

    //12/7/13, removed an erroneous factor of -1
    return ( 1.0/( (CL_TYPE)n - 1.0) )*( pow( cos(upper_limit), -1.0*((CL_TYPE)n) + 1.0 ) - pow( cos(lower_limit), -1.0*((CL_TYPE)n) + 1.0)  );
}

//______________________________________________________________________________
void
I_cheb1_array(int l_max, CL_TYPE lower_limit, CL_TYPE upper_limit,  CL_TYPE2* val)
{
    val[0].s0 = I_secn( 2, lower_limit, upper_limit); //(0,0)

    int si, si_m1, si_a, si_b;

    for(int j = 1; j <= l_max; j++)
    {
        si = j*(j+1)/2;
        si_m1 = j*(j-1)/2;
        val[si].s0 = I_secn( j+2, lower_limit, upper_limit);   //(j,0) base case
        val[si + 1].s0 = val[si_m1].s0; //(j,1) base case

        for(int k=2; k <=j; k++)
        {
            si = j*(j+1)/2 + k;
            si_a = j*(j-1)/2 + k - 1;
            si_b = j*(j+1)/2 + k - 2;
            val[si].s0 = 2.0*val[si_a].s0 - val[si_b].s0;
        }
    }
}


//______________________________________________________________________________

void
I_cheb2_array(int l_max, CL_TYPE lower_limit, CL_TYPE upper_limit,  CL_TYPE2* val)
{
    val[0].s1 = 0.0; //(0,0)

    int si, si_a, si_b;

    for(int j = 1; j <= l_max; j++)
    {
        si = j*(j+1)/2;
        val[si].s1 = 0; //(j,0) base case
        val[si + 1].s1 = I_trig1(j+2, lower_limit, upper_limit); //(j,1) base case

        for(int k=2; k <=j; k++)
        {
            si = j*(j+1)/2 + k;
            si_a = j*(j-1)/2 + k - 1;
            si_b = j*(j+1)/2 + k - 2;
            val[si].s1 = 2.0*val[si_a].s1 - val[si_b].s1;
        }
    }
}




//______________________________________________________________________________
//normalize an array of I_cheb integrals

void
K_normalize_array(int l_max, CL_TYPE h, __constant const CL_TYPE* plm,  CL_TYPE2* val)
{
    CL_TYPE hpow = h;
    CL_TYPE fac;
    int si;

    for(int l=0; l<=l_max; l++)
    {
        hpow *= h;
        fac = (1.0/( (CL_TYPE)l + 2.0) );
        for(int m=0; m <= l; m++)
        {
            si = (l*(l+1))/2 + m;
            val[si].s0 *= fac*hpow*plm[si];
            val[si].s1 *= fac*hpow*plm[si];
        }
    }
}

#endif /* KFMSphericalMultipole_Defined_H */
