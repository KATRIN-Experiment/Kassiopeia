#ifndef KGAUSSLEGENDREQUADRATURE_DEF
#define KGAUSSLEGENDREQUADRATURE_DEF


namespace KEMField
{
	struct KGaussLegendreQuadrature
	{
		void operator() (double (*f)(double),
				double a,
				double b,
				unsigned int n,
				double *ret );

	};
}

#endif /* KGAUSSLEGENDREQUADRATURE_DEF */
