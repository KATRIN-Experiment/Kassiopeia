#ifndef KFUNDAMENTALTYPES_DEF
#define KFUNDAMENTALTYPES_DEF

#include "KTypelist.hh"

#include <string>

namespace KEMField
{
  typedef KTYPELIST_14( bool, char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double , std::string ) FundamentalTypes;

  extern const std::string FundamentalTypeNames[14];
}

#endif /* KFUNDAMENTALTYPES_DEF */
