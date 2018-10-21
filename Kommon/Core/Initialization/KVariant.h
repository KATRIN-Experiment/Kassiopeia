// KVariant.h //
// Author: Sanshiro Enomoto <sanshiro@uw.edu> //


#ifndef KVariant_h__
#define KVariant_h__

#include <string>
#include <iostream>
#include <sstream>
#include "KException.h"


// "variant" is not "any" as in boost, because it does not support
// general types as a template, but limits to primitive types.
// In return, variant provides non-explicit constructors and
// type-cast operators which make conversions invisible. 
// (If you do this with a template, it will generate a global
// type conversion path between any types, and can destroy the
// C++ conversion rules.)
//
// "variant" uses "union" instead of dynamically allocated templated 
// type holder object (like "any" in boost), which will be an advantage 
// for large scale data table etc. 


namespace katrin {


class KUnknown {
  public:
    // use RTTI to get actual type //
    KUnknown(void) {}
    virtual ~KUnknown() {}
    virtual KUnknown* Clone(void) const = 0;
};


/**
 * \brief Variant data type (union + data conversion interface)
 */
class KVariant {
  public:
    inline KVariant(void);
    inline KVariant(bool Value);
    inline KVariant(int Value);
    inline KVariant(unsigned int Value);
    inline KVariant(long Value);
    inline KVariant(unsigned long Value);
    inline KVariant(long long Value);
    inline KVariant(unsigned long long Value);
    inline KVariant(float Value);
    inline KVariant(double Value);
    inline KVariant(const std::string& Value);
    inline KVariant(const char* Value);
    inline KVariant(const KUnknown& Value);
    inline KVariant(const KVariant& Value);
    inline ~KVariant();
    inline KVariant& operator=(const KVariant& Value);
    inline void Assign(const KVariant& Value) ;
  public:
    template<typename T> inline T As(void) const ;
    inline KVariant Or(const KVariant& DefaultValue) const;
    inline operator bool() const ;
    inline operator int() const ;
    inline operator unsigned int() const ;
    inline operator long() const ;
    inline operator unsigned long() const ;
    inline operator long long() const ;
    inline operator unsigned long long() const ;
    inline operator float() const ;
    inline operator double() const ;
    inline operator std::string() const;
    inline operator const KUnknown&() const ;
  public:
    inline bool IsVoid(void) const;
    inline bool IsBool(void) const;
    inline bool IsInteger(void) const;
    inline bool IsNumeric(void) const;
    inline bool IsString(void) const;
    inline bool IsUnknown(void) const;
    bool AsBool(void) const ;
    long long AsLong(void) const ;
    double AsDouble(void) const ;
    std::string AsString(void) const;
    KUnknown& AsUnknown(void) ;
    const KUnknown& AsUnknown(void) const ;
  private:
    enum TValueType { 
	Type_Void, Type_Bool, Type_Long, Type_Double, Type_String, Type_Unknown
    } fType;
    union TPrimitive {
        bool fBoolValue;
        long long fLongValue;
        double fDoubleValue;
        std::string* fStringValue;
        KUnknown* fUnknownValue;
    } fPrimitive;
};



template<typename T>
struct KVariantDecoder {
  // explicit partial instantiations only //
  private:
    static T As(const KVariant&) { return T(); }
};

template<> struct KVariantDecoder<void> {
    static void As(const KVariant&) { }
};

template<> struct KVariantDecoder<bool> {
    static bool As(const KVariant& Value) {
        return Value.AsBool();
    }
};

template<> struct KVariantDecoder<signed char> {
    static signed char As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<unsigned char> {
    static unsigned char As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<short> {
    static short As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<unsigned short> {
    static unsigned short As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<int> {
    static int As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<unsigned int> {
    static unsigned int As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<long> {
    static long As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<unsigned long> {
    static unsigned long As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<long long> {
    static long long As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<unsigned long long> {
    static unsigned long long As(const KVariant& Value) {
        return Value.AsLong();
    }
};

template<> struct KVariantDecoder<float> {
    static float As(const KVariant& Value) {
        return Value.AsDouble();
    }
};

template<> struct KVariantDecoder<double> {
    static double As(const KVariant& Value) {
        return Value.AsDouble();
    }
};

template<> struct KVariantDecoder<std::string> {
    static std::string As(const KVariant& Value) {
        return Value.AsString();
    }
};

template<> struct KVariantDecoder<const char*> {
    static std::string As(const KVariant& Value) {
        return Value.AsString().c_str();
    }
};

template<> struct KVariantDecoder<const KUnknown&> {
    static const KUnknown& As(const KVariant& Value) {
        return Value.AsUnknown();
    }
};



KVariant::KVariant(void)
: fType(Type_Void)
{
}

KVariant::KVariant(bool Value)
: fType(Type_Bool)
{
    fPrimitive.fBoolValue = Value;
}

KVariant::KVariant(int Value)
: fType(Type_Long)
{
    fPrimitive.fLongValue = Value;
}

KVariant::KVariant(unsigned int Value)
: fType(Type_Long)
{
    fPrimitive.fLongValue = Value;
}

KVariant::KVariant(long Value)
: fType(Type_Long)
{
    fPrimitive.fLongValue = Value;
}

KVariant::KVariant(unsigned long Value)
: fType(Type_Long)
{
    fPrimitive.fLongValue = Value;
}

KVariant::KVariant(long long Value)
: fType(Type_Long)
{
    fPrimitive.fLongValue = Value;
}

KVariant::KVariant(unsigned long long Value)
: fType(Type_Long)
{
    fPrimitive.fLongValue = Value;
}

KVariant::KVariant(float Value)
: fType(Type_Double)
{
    fPrimitive.fDoubleValue = Value;
}

KVariant::KVariant(double Value)
: fType(Type_Double)
{
    fPrimitive.fDoubleValue = Value;
}

KVariant::KVariant(const std::string& Value)
: fType(Type_String)
{
    fPrimitive.fStringValue = new std::string(Value);
}

KVariant::KVariant(const char* Value)
: fType(Type_String)
{
    fPrimitive.fStringValue = new std::string(Value);
}

KVariant::KVariant(const KUnknown& Value)
: fType(Type_Unknown)
{
    fPrimitive.fUnknownValue = Value.Clone();
}

KVariant::KVariant(const KVariant& Value)
: fType(Value.fType), fPrimitive(Value.fPrimitive)
{
    if (fType == Type_String) {
	fPrimitive.fStringValue = (
	    new std::string(*Value.fPrimitive.fStringValue)
	);
    }
    else if (fType == Type_Unknown) {
	fPrimitive.fUnknownValue = Value.fPrimitive.fUnknownValue->Clone();
    }
}

KVariant::~KVariant()
{
    if (fType == Type_String) {
	delete fPrimitive.fStringValue;
    }
    else if (fType == Type_Unknown) {
	delete fPrimitive.fUnknownValue;
    }
}

KVariant& KVariant::operator=(const KVariant& Value)
{
    if (&Value == this) {
	return *this;
    }

    if (fType == Type_String) {
	delete fPrimitive.fStringValue;
    }
    else if (fType == Type_Unknown) {
	delete fPrimitive.fUnknownValue;
    }

    if (Value.fType == Type_String) {
	fPrimitive.fStringValue = (
	    new std::string(*Value.fPrimitive.fStringValue)
	);
    }
    else if (Value.fType == Type_Unknown) {
	fPrimitive.fUnknownValue = Value.fPrimitive.fUnknownValue->Clone();
    }
    else {
	fPrimitive = Value.fPrimitive;
    }
    fType = Value.fType;

    return *this;
}

void KVariant::Assign(const KVariant& Value) 
{
    if (fType == Type_Void) {
	throw KException() << "assignment to void variable not allowed";
    }
    else if (fType == Type_Bool) {
        fPrimitive.fBoolValue = Value.AsBool();
    }
    else if (fType == Type_Long) {
        fPrimitive.fLongValue = Value.AsLong();
    }
    else if (fType == Type_Double) {
        fPrimitive.fDoubleValue = Value.AsDouble();
    }
    else if (fType == Type_String) {
        *(fPrimitive.fStringValue) = Value.AsString();
    }
    else {
	throw KException() << "assignment to Unknown variable not allowed";
    }
}

template <typename T> 
inline T KVariant::As(void) const 
{
    return katrin::KVariantDecoder<T>::As(*this);
}

inline KVariant KVariant::Or(const KVariant& DefaultValue) const
{
    if (fType == Type_Void) {
        return DefaultValue;
    }
    else {
        return *this;
    }
}

KVariant::operator bool() const 
{
    return As<bool>();
}

KVariant::operator int() const 
{
    return As<int>();
}

KVariant::operator unsigned int() const 
{
    return As<unsigned int>();
}

KVariant::operator long() const 
{
    return As<long>();
}

KVariant::operator unsigned long() const 
{
    return As<unsigned long>();
}

KVariant::operator long long() const 
{
    return As<long long>();
}

KVariant::operator unsigned long long() const 
{
    return As<unsigned long long>();
}

KVariant::operator float() const 
{
    return As<float>();
}

KVariant::operator double() const 
{
    return As<double>();
}

KVariant::operator std::string() const
{
    return As<std::string>();
}

KVariant::operator const KUnknown&() const 
{
    return As<const KUnknown&>();
}

bool KVariant::IsVoid(void) const
{
    return (fType == Type_Void);
}

bool KVariant::IsBool(void) const
{
    return (fType == Type_Bool);
}

bool KVariant::IsInteger(void) const
{
    return (fType == Type_Long);
}

bool KVariant::IsNumeric(void) const
{
    return ((fType == Type_Long) || (fType == Type_Double));
}

bool KVariant::IsString(void) const
{
    return (fType == Type_String);
}

bool KVariant::IsUnknown(void) const
{
    return (fType == Type_Unknown);
}
    


inline std::ostream& operator<<(std::ostream& os, const KVariant& Value) 
{
    if (Value.IsInteger()) {
	os << (long) Value;
    }
    else if (Value.IsNumeric()) {
	os << (double) Value;
    }
    else {
	os << (const std::string&) Value;
    }

    return os;
}



#if 1
// do not use const KVariant& for the following operator overloading
// otherwise the operators will be ambiguous (e.g., enum + int)

template <typename T>
inline KVariant operator+(KVariant& Left, const T& Right)
{
    if (Left.IsInteger()) {
        return KVariant(Left.As<long long>() + Right);
    }
    else if (Left.IsNumeric()) {
        return KVariant(Left.As<double>() + Right);
    }
    else if (Left.IsString()) {
        return KVariant(
            Left.As<std::string>() + KVariant(Right).As<std::string>()
        );
    }

    return KVariant(Left.As<T>() + Right);
}

template <typename T>
inline KVariant operator+(const T& Left, KVariant& Right)
{
    if (Right.IsInteger()) {
        return KVariant(Left + Right.As<long long>());
    }
    else if (Right.IsNumeric()) {
        return KVariant(Left + Right.As<double>());
    }
    else if (Right.IsString()) {
        return KVariant(
            KVariant(Left).As<std::string>() + Right.As<std::string>()
        );
    }

    return KVariant(Left + Right.As<T>());
}

inline KVariant operator+(KVariant& Left, KVariant& Right)
{
    if (Right.IsInteger()) {
        return operator+(Left, Right.As<long long>());
    }
    else if (Right.IsNumeric()) {
        return operator+(Left, Right.As<double>());
    }
    else {
        return KVariant(Left.As<std::string>() + Right.As<std::string>());
    }
}

inline KVariant operator+(KVariant& Left, const std::string& Right) 
{
    return KVariant(Left.As<std::string>() + Right);
}

inline KVariant operator+(const std::string& Left, KVariant& Right)
{
    return KVariant(Left + Right.As<std::string>());
}

inline KVariant operator+(KVariant& Left, const char Right[]) 
{
    return KVariant(Left.As<std::string>() + std::string(Right));
}

inline KVariant operator+(const char Left[], KVariant& Right)
{
    return KVariant(std::string(Left) + Right.As<std::string>());
}

template <typename T>
inline KVariant& operator+=(KVariant& This, const T& Value)
{
    return This = operator+(This, Value);
}

template <typename T>
inline KVariant& operator-=(KVariant& This, const T& Value)
{
    if (This.IsInteger()) {
        return This = This.As<long long>() - Value;
    }
    else if (This.IsNumeric()) {
        return This = This.As<double>() - Value;
    }

    return This = This.As<T>() - Value;
}

template <typename T>
inline KVariant& operator*=(KVariant& This, const T& Value)
{
    if (This.IsInteger()) {
        return This = This.As<long long>() * Value;
    }
    else if (This.IsNumeric()) {
        return This = This.As<double>() * Value;
    }

    return This = This.As<T>() * Value;
}

template <typename T>
inline KVariant& operator/=(KVariant& This, const T& Value)
{
    if (This.IsInteger()) {
        return This = This.As<long long>() / Value;
    }
    else if (This.IsNumeric()) {
        return This = This.As<double>() / Value;
    }

    return This = This.As<T>() / Value;
}

template <typename T>
inline bool operator==(KVariant& This, const T& Value)
{
    return This.As<T>() == Value;
}

inline bool operator==(KVariant& This, const char Value[])
{
    return This.As<std::string>() == std::string(Value);
}

template <typename T>
inline bool operator!=(KVariant& This, const T& Value)
{
    return This.As<T>() != Value;
}

inline bool operator!=(KVariant& This, const char Value[])
{
    return This.As<std::string>() != std::string(Value);
}

template <typename T>
inline bool operator>(KVariant& This, const T& Value)
{
    return This.As<T>() > Value;
}

template <typename T>
inline bool operator<(KVariant& This, const T& Value)
{
    return This.As<T>() < Value;
}

template <typename T>
inline bool operator>=(KVariant& This, const T& Value)
{
    return This.As<T>() >= Value;
}

template <typename T>
inline bool operator<=(KVariant& This, const T& Value)
{
    return This.As<T>() <= Value;
}

#endif


typedef KVariant var;


}
#endif
