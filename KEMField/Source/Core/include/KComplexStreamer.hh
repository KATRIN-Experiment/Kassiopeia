#ifndef KCOMPLEXSTREAMER_DEF
#define KCOMPLEXSTREAMER_DEF

#include <complex>

namespace KEMField
{
template<typename Type, typename Stream> Stream& operator>>(Stream& s, std::complex<Type>& c)
{
    Type real, imag;
    s >> real;
    s >> imag;
    c = std::complex<Type>(real, imag);
    return s;
}

template<typename Type, typename Stream> Stream& operator<<(Stream& s, const std::complex<Type>& c)
{
    s << c.real();
    s << c.imag();
    return s;
}

}  // namespace KEMField

#endif /* KCOMPLEXSTREAMER_DEF */
