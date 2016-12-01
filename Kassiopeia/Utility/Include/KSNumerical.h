#ifndef Kassiopeia_KSNumerical_h_
#define Kassiopeia_KSNumerical_h_

#include <limits>

namespace Kassiopeia
{

    template< class XType >
    struct KSNumerical
    {
        static constexpr XType Maximum() { return std::numeric_limits<XType>::max(); }
        static constexpr XType Zero()    { return 0; }
        static constexpr XType Minimum() { return std::numeric_limits<XType>::min(); }
	static constexpr XType Lowest()  { return std::numeric_limits<XType>::lowest(); } 
    };

    template< >
    struct KSNumerical< bool >
    {
        static constexpr bool Maximum() { return true; }
        static constexpr bool Zero()    { return false; }
        static constexpr bool Minimum() { return false; }
	static constexpr bool Lowest()  { return false; }
    };

}

#endif
