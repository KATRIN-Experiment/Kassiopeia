#ifndef KEXCEPTION_H_
#define KEXCEPTION_H_

#include <exception>
#include <sstream>
#include <string>

namespace katrin
{

/**
 * Base class for exception objects
 *
 * @headerfile KException.h
 * @author Daniel Furse <daniel.furse@gmail.com>
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 *
 * Base class for all exceptions thrown by katrin code.
 */
class KExceptionBase : public std::exception
{
  public:
    KExceptionBase();
    ~KExceptionBase() throw() override {}

    KExceptionBase(const KExceptionBase& toCopy);
    void operator=(const KExceptionBase& toCopy);

    const char* what() const throw() override;

  protected:
    std::ostringstream fMessage;
    std::string fNestedMessage;

    mutable std::string fWhat;
};

class KException;

template<class XDerivedType, class XBaseType = KException> class KExceptionPrototype : public XBaseType
{
  public:
    template<class XValue> inline XDerivedType& operator<<(const XValue& toAppend)
    {
        return this->Append(toAppend);
    }
    template<class XValue> inline XDerivedType& Append(const XValue& toAppend)
    {
        XBaseType::fMessage << toAppend;
        return static_cast<XDerivedType&>(*this);
    }

    inline XDerivedType& operator<<(const std::exception& toNest)
    {
        return this->Nest(toNest);
    }

    inline XDerivedType& Nest(const std::exception& toNest)
    {
        XBaseType::fNestedMessage = toNest.what();
        return static_cast<XDerivedType&>(*this);
    }
};


class KException : public KExceptionPrototype<KException, KExceptionBase>
{};


inline KExceptionBase::KExceptionBase() {}

inline KExceptionBase::KExceptionBase(const KExceptionBase& toCopy) :
    std::exception(toCopy),
    fMessage(toCopy.fMessage.str()),
    fNestedMessage(toCopy.fNestedMessage)
{}

inline void KExceptionBase::operator=(const KExceptionBase& toCopy)
{
    std::exception::operator=(toCopy);
    fMessage.str(toCopy.fMessage.str());
    fNestedMessage = toCopy.fNestedMessage;
}

inline const char* KExceptionBase::what() const throw()
{
    fWhat = fMessage.str();
    if (!fNestedMessage.empty()) {
        fWhat.append(" [" + fNestedMessage + "]");
    }
    return fWhat.c_str();
}

template<class XDerivedType, class XBaseType>
inline std::ostream& operator<<(std::ostream& os, const KExceptionPrototype<XDerivedType, XBaseType>& e)
{
    return os << e.what();
}

}  // namespace katrin

#endif  // KEXCEPTION_H
