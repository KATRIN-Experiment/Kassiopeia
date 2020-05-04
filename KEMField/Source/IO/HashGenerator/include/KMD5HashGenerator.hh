#ifndef KMD5HASHGENERATOR_DEF
#define KMD5HASHGENERATOR_DEF

#include "KFundamentalTypes.hh"
#include "KTypeManipulation.hh"
#include "md5.hh"

#include <cmath>
#include <iomanip>
#include <limits>
#include <set>
#include <string>
#include <typeinfo>

namespace KEMField
{

/**
* @class KMD5HashGenerator
*
* @brief A streamer class for generating MD5 Hashes.
*
* KMD5HashGenerator is a class that accepts data streams, and generates MD5
* hashes from them.
*
* @author T.J. Corona
*/

class KMD5HashGenerator;

template<typename Type> struct KMD5HashGeneratorType
{
    friend inline KMD5HashGenerator& operator<<(KMD5HashGeneratorType<Type>& d, const Type& x)
    {
        if (!(d.Omit())) {
            auto& y = const_cast<Type&>(x);
            d.GetMD5().update(reinterpret_cast<unsigned char*>(&y), sizeof(Type));
        }
        return d.Self();
    }
    virtual ~KMD5HashGeneratorType() {}
    virtual bool Omit() const = 0;
    virtual MD5& GetMD5() = 0;
    virtual KMD5HashGenerator& Self() = 0;
};

template<> struct KMD5HashGeneratorType<float>
{
    friend inline KMD5HashGenerator& operator<<(KMD5HashGeneratorType<float>& d, const float& x)
    {
        static int one = 1;
        static int endian_min = (*(char*) &one == 1 ? 0 : sizeof(float) - 1);
        static int endian_max = (*(char*) &one == 1 ? sizeof(float) - 1 : 0);
        static int endian_dir = (*(char*) &one == 1 ? 1 : -1);

        if (!(d.Omit())) {
            // we cannot resolve values near machine precision, so we omit them
            if (fabs(x) < d.Threshold())
                return d.Self();

            float y = x;

            // first we round
            if (d.MaskedBits() || d.MaskedBytes()) {
                unsigned short index, significantBit;
                if (d.MaskedBits()) {
                    index = endian_min + endian_dir * d.MaskedBytes();
#if (__GNUC__ > 5) || (__clang_major__ == 3) && (__clang_minor__ >= 8) || (__clang_major__ > 3)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-negative-value"
#endif
                    significantBit = (~(0xff << 1)) << (d.MaskedBits() - 1);
#if (__GNUC__ > 5) || (__clang_major__ == 3) && (__clang_minor__ >= 8) || (__clang_major__ > 3)
#pragma GCC diagnostic pop
#endif
                }
                else {
                    index = endian_min + endian_dir * (d.MaskedBytes() - 1);
                    significantBit = 0x80;
                }

                if (reinterpret_cast<unsigned char*>(&y)[index] & significantBit) {
                    double roundOff = y;
                    reinterpret_cast<unsigned char*>(&roundOff)[index] ^= significantBit;
                    y += (y - roundOff);
                }
            }

            // then we mask the volatile bytes
            for (int i = endian_min; i != endian_max; i += endian_dir) {
                if (i == d.MaskedBytes()) {
                    reinterpret_cast<unsigned char*>(&y)[i] &= (0xff << d.MaskedBits());
                    break;
                }
                reinterpret_cast<unsigned char*>(&y)[i] &= 0x00;
            }
            d.GetMD5().update(reinterpret_cast<unsigned char*>(&y), sizeof(float));
        }
        return d.Self();
    }

    virtual ~KMD5HashGeneratorType() {}
    virtual bool Omit() const = 0;
    virtual unsigned short MaskedBits() const = 0;
    virtual unsigned short MaskedBytes() const = 0;
    virtual double Threshold() const = 0;
    virtual MD5& GetMD5() = 0;
    virtual KMD5HashGenerator& Self() = 0;
};

template<> struct KMD5HashGeneratorType<double>
{
    friend inline KMD5HashGenerator& operator<<(KMD5HashGeneratorType<double>& d, const double& x)
    {
        static int one = 1;
        static int endian_min = (*(char*) &one == 1 ? 0 : sizeof(double) - 1);
        static int endian_max = (*(char*) &one == 1 ? sizeof(double) - 1 : 0);
        static int endian_dir = (*(char*) &one == 1 ? 1 : -1);

        if (!(d.Omit())) {
            // we cannot resolve values near machine precision, so we omit them
            if (fabs(x) < d.Threshold())
                return d.Self();

            double y = x;

            // first we round
            if (d.MaskedBits() || d.MaskedBytes()) {
                unsigned short index, significantBit;
                if (d.MaskedBits()) {
                    index = endian_min + endian_dir * d.MaskedBytes();
#if (__GNUC__ > 5) || (__clang_major__ == 3) && (__clang_minor__ >= 8) || (__clang_major__ > 3)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-negative-value"
#endif
                    significantBit = (~(0xff << 1)) << (d.MaskedBits() - 1);
#if (__GNUC__ > 5) || (__clang_major__ == 3) && (__clang_minor__ >= 8) || (__clang_major__ > 3)
#pragma GCC diagnostic pop
#endif
                }
                else {
                    index = endian_min + endian_dir * (d.MaskedBytes() - 1);
                    significantBit = 0x80;
                }

                if (reinterpret_cast<unsigned char*>(&y)[index] & significantBit) {
                    double roundOff = y;
                    reinterpret_cast<unsigned char*>(&roundOff)[index] ^= significantBit;
                    y += (y - roundOff);
                }
            }

            // then we mask the volatile bytes
            for (int i = endian_min; i != endian_max; i += endian_dir) {
                if (i == d.MaskedBytes()) {
                    reinterpret_cast<unsigned char*>(&y)[i] &= (0xff << d.MaskedBits());
                    break;
                }
                reinterpret_cast<unsigned char*>(&y)[i] &= 0x00;
            }
            d.GetMD5().update(reinterpret_cast<unsigned char*>(&y), sizeof(double));
        }
        return d.Self();
    }

    virtual ~KMD5HashGeneratorType() {}
    virtual bool Omit() const = 0;
    virtual unsigned short MaskedBits() const = 0;
    virtual unsigned short MaskedBytes() const = 0;
    virtual double Threshold() const = 0;
    virtual MD5& GetMD5() = 0;
    virtual KMD5HashGenerator& Self() = 0;
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KMD5HashGeneratorType> KMD5HashGeneratorFundamentalTypes;

class KMD5HashGenerator : public KMD5HashGeneratorFundamentalTypes
{
  public:
    KMD5HashGenerator() : fOmit(false), fMaskedBits(1), fThreshold(1.e-12), fMD5(nullptr), fOmitting(nullptr) {}
    ~KMD5HashGenerator() override
    {
        if (fMD5)
            delete fMD5;
    }

    template<class X> void Omit(Type2Type<X>)
    {
        fOmitted.insert(&typeid(X));
    }

    template<class Head, class Tail> void Omit(KTypelist<Head, Tail>);

    void Omit(KNullType) {}

    template<class X> std::string GenerateHash(const X& x);

    std::string GenerateHashFromString(const std::string& x)
    {
        unsigned int ch_size = x.size() + 1;
        char* ch_array;
        ch_array = new char[ch_size];
        for (unsigned int i = 0; i < x.size(); i++) {
            ch_array[i] = x.at(i);
        }
        ch_array[x.size()] = '\0';

        if (fMD5)
            delete fMD5;
        fMD5 = new MD5();
        fMD5->update(reinterpret_cast<unsigned char*>(ch_array), ch_size);
        fMD5->finalize();
        std::string value(fMD5->hex_digest());
        delete[] ch_array;

        return value;
    }

    template<class Streamed> void PreStreamOutAction(const Streamed&);
    template<class Streamed> void PostStreamOutAction(const Streamed&);

    void MaskedBits(unsigned short i)
    {
        fMaskedBits = i;
    }
    void MaskedBytes(unsigned short i)
    {
        fMaskedBits = 8 * i;
    }
    unsigned short MaskedBits() const override
    {
        return fMaskedBits % 8;
    }
    unsigned short MaskedBytes() const override
    {
        return fMaskedBits / 8;
    }
    void Threshold(double d)
    {
        fThreshold = d;
    }
    double Threshold() const override
    {
        return fThreshold;
    }

  protected:
    bool Omit() const override
    {
        return fOmit;
    }
    class MD5& GetMD5() override
    {
        return *fMD5;
    }
    KMD5HashGenerator& Self() override
    {
        return *this;
    }

    bool fOmit;
    unsigned short fMaskedBits;
    double fThreshold;
    class MD5* fMD5;
    const std::type_info* fOmitting;

    std::set<const std::type_info*> fOmitted;
};

template<class Head, class Tail> void KMD5HashGenerator::Omit(KTypelist<Head, Tail>)
{
    Omit(Type2Type<Head>());
    return Omit(Tail());
}

template<class X> std::string KMD5HashGenerator::GenerateHash(const X& x)
{
    if (fMD5)
        delete fMD5;
    fMD5 = new MD5();
    *this << x;
    fMD5->finalize();
    std::string value(fMD5->hex_digest());
    return value;
}

template<class Streamed> void KMD5HashGenerator::PreStreamOutAction(const Streamed&)
{
    for (auto it = fOmitted.begin(); it != fOmitted.end(); ++it) {
        if (*(*it) == typeid(Streamed)) {
            fOmitting = &typeid(Streamed);
            fOmit = true;
        }
    }
}

template<class Streamed> void KMD5HashGenerator::PostStreamOutAction(const Streamed&)
{
    if (fOmitting) {
        if (typeid(Streamed) == *fOmitting)
            fOmit = false;
    }
}
}  // namespace KEMField

#endif /* KMD5HASHGENERATOR_DEF */
