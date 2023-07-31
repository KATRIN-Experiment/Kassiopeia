// KVariant.cxx //
// Author: Sanshiro Enomoto <sanshiro@uw.edu> //

#include "KVariant.h"

#include <limits>
#include <string>


//#define USE_KBASE_STRING_UTILS 1
#ifdef USE_KBASE_STRING_UTILS
#include "KBaseStringUtils.h"
#else
#include <cstring>
#endif

using namespace std;
using namespace katrin;


bool KVariant::AsBool() const
{
    if (fType == Type_Void) {
        throw KException() << "conversion from undefined to bool";
    }
    else if (fType == Type_Bool) {
        return fPrimitive.fBoolValue;
    }
    if (fType == Type_Long) {
        return fPrimitive.fLongValue != 0;
    }
    else if (fType == Type_Double) {
        return fPrimitive.fDoubleValue != 0;
    }
    else if (fType == Type_String) {
#ifdef USE_KBASE_STRING_UTILS
        return KBaseStringUtils::Convert<bool>(*fPrimitive.fStringValue);
#else
        //... TODO: implement YAML-compatible type rules??? Maybe not??? ...//
        if (*fPrimitive.fStringValue == "true") {
            return true;
        }
        else if (*fPrimitive.fStringValue == "false") {
            return false;
        }
#endif
    }
    else if (fType == Type_Unknown) {
        return fPrimitive.fUnknownValue != nullptr;
    }

    throw KException() << "bad type to convert to bool: \"" << AsString() << "\"";
}

long long KVariant::AsLong() const
{
    if (fType == Type_Void) {
        throw KException() << "conversion from undefined to integer";
    }
    else if (fType == Type_Bool) {
        return fPrimitive.fBoolValue ? 1 : 0;
    }
    else if (fType == Type_Long) {
        return fPrimitive.fLongValue;
    }
    else if (fType == Type_Double) {
        return (long long) fPrimitive.fDoubleValue;
    }
    else if (fType == Type_String) {
#ifdef USE_KBASE_STRING_UTILS
        return KBaseStringUtils::Convert<long>(*fPrimitive.fStringValue);
#else
        long Value;
        const char* Start = fPrimitive.fStringValue->c_str();
        char* End;
        errno = 0;
        if (strncmp(Start, "0x", 2) == 0) {
            Value = strtol(Start, &End, 0);
        }
        else {
            Value = strtol(Start, &End, 10);
        }
        if (((Value == 0) && (errno != 0)) || (*End != '\0')) {
            throw KException() << "bad string to convert to int: \"" << *fPrimitive.fStringValue << "\"";
        }
        return Value;
#endif
    }

    throw KException() << "bad type to convert to int: \"" << AsString() << "\"";
}

double KVariant::AsDouble() const
{
    if (fType == Type_Void) {
        //throw KException() << "conversion from undefined to double";
        return std::numeric_limits<double>::quiet_NaN();
    }
    else if (fType == Type_Bool) {
        return fPrimitive.fBoolValue ? 1 : 0;
    }
    else if (fType == Type_Long) {
        return (double) fPrimitive.fLongValue;
    }
    else if (fType == Type_Double) {
        return fPrimitive.fDoubleValue;
    }
    else if (fType == Type_String) {
#ifdef USE_KBASE_STRING_UTILS
        return KBaseStringUtils::Convert<double>(*fPrimitive.fStringValue);
#else
        double Value;
        const char* Start = fPrimitive.fStringValue->c_str();
        char* End;
        errno = 0;
        Value = strtod(Start, &End);
        if (((Value == 0) && (errno != 0)) || (*End != '\0')) {
            throw KException() << "bad string to convert to double: \"" << *fPrimitive.fStringValue << "\"";
        }
        return Value;
#endif
    }

    throw KException() << "bad type to convert to int: \"" << AsString() << "\"";
}

std::string KVariant::AsString(int precision) const
{
    if (fType == Type_String) {
        return *fPrimitive.fStringValue;
    }
    if (fType == Type_Void) {
        return "";
    }

    std::ostringstream os;
    if (fType == Type_Bool) {
        os << (fPrimitive.fBoolValue ? "true" : "false");
    }
    else if (fType == Type_Long) {
        os << fPrimitive.fLongValue;
    }
    else if (fType == Type_Double) {
        if (precision > 0) {
            auto prev_precision = os.precision(precision);
            os << fPrimitive.fDoubleValue;
            os.precision(prev_precision);
        }
        else {
            os << fPrimitive.fDoubleValue;
        }
    }
    else {
        os << "Unknown@" << this;
    }

    return os.str();
}

KUnknown& KVariant::AsUnknown()
{
    if (fType == Type_Void) {
        throw KException() << "conversion from undefined to unknown";
    }
    else if (fType != Type_Unknown) {
        throw KException() << "bad type to convert to 'unknown': " << AsString();
    }

    return *fPrimitive.fUnknownValue;
}

const KUnknown& KVariant::AsUnknown() const
{
    if (fType == Type_Void) {
        throw KException() << "conversion from undefined to unknown";
    }
    else if (fType != Type_Unknown) {
        throw KException() << "bad type to convert to 'unknown': " << AsString();
    }

    return *fPrimitive.fUnknownValue;
}

std::map<int, KVariant> KVariant::SplitBy(const std::string& Separator, std::vector<KVariant> DefaultValueList) const
{
    map<int, KVariant> Result;
    for (unsigned i = 0; i < DefaultValueList.size(); i++) {
        Result[i] = DefaultValueList[i];
    }

    std::string str = this->As<std::string>();
    for (unsigned i = 0; !str.empty(); i++) {
        auto pos = str.find(Separator);
        std::string v = str.substr(0, pos);
        if (!v.empty()) {
            Result[i] = KVariant(v);
        }
        if (pos == string::npos) {
            break;
        }
        str = str.substr(pos + Separator.size());
    }

    return Result;
}
