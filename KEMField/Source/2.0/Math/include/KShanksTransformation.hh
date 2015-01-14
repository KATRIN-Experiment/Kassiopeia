#ifndef KSHANKSTRANSFORMATION_DEF
#define KSHANKSTRANSFORMATION_DEF

namespace KEMField
{

/**
* @class KShanksTransformation
*
* @brief A functor for applying the Shanks transformation
*
* @author T.J. Corona
*/

  struct KShanksTransformation
  {
    double operator() (double A_0,double A_1,double A_2) const
    {
      return (A_2*A_0 - A_1*A_1)/(A_2 - 2.*A_1 + A_0);
    }
  };
}

#endif /* KSHANKSTRANSFORMATION_DEF */
