#ifndef KSAPODConverter_HH__
#define KSAPODConverter_HH__

#include "KSATokenizer.hh"

#include <iomanip>
#include <limits>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace KEMField
{


/**
*
*@file KSAPODConverter.hh
*@class KSAPODConverter
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Dec 26 10:50:36 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


//here U must be a plain old data (POD) type, such as int, double, etc.

template<typename U> class KSAPODConverter
{
  public:
    KSAPODConverter()
    {
        ;
    }
    virtual ~KSAPODConverter()
    {
        ;
    };

    void ConvertParameterToString(std::string& string_value, const U& value)
    {
        fStream = new std::stringstream();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        fStream->str("");
        fStream->clear();

        *fStream << value;
        //            *fStream << LINE_DELIM;

        string_value = fStream->str();

        delete fStream;
    }

    void ConvertParameterToString(std::string& string_value, const U* value)
    {
        fStream = new std::stringstream();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        fStream->str("");
        fStream->clear();

        *fStream << *value;
        //            *fStream << LINE_DELIM;

        string_value = fStream->str();

        delete fStream;
    }


    void ConvertStringToParameter(const std::string& string_value, U& value)
    {
        fStream = new std::stringstream();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        fStream->str("");
        fStream->clear();

        *fStream << string_value;

        *fStream >> value;

        delete fStream;
    };


  protected:
    std::stringstream* fStream;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//here is the partial specialization for vectors of PODs


template<typename U> class KSAPODConverter<std::vector<U>>
{
  public:
    KSAPODConverter()
    {
        fTokenizer = new KSATokenizer();
        fTokenizer->SetDelimiter(ELEM_DELIM);
        fTokens.clear();
    }

    virtual ~KSAPODConverter()
    {
        delete fTokenizer;
    };

    void ConvertParameterToString(std::string& string_value, const std::vector<U>& value)
    {

        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        //loop over vector exporting the values to the stringstream
        for (unsigned int i = 0; i < value.size(); i++) {
            *fStream << value[i];
            if (i != value.size() - 1) {
                *fStream << ELEM_DELIM;
            }
        }

        //            *fStream << LINE_DELIM;

        string_value = fStream->str();

        delete fStream;
    }

    void ConvertParameterToString(std::string& string_value, const std::vector<U>* value)
    {
        //assumes SetValue has just been called
        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        //loop over vector exporting the values to the stringstream
        for (unsigned int i = 0; i < value->size(); i++) {
            *fStream << (*value)[i];
            if (i != value->size() - 1) {
                *fStream << ELEM_DELIM;
            }
        }

        //            *fStream << LINE_DELIM;

        string_value = fStream->str();

        delete fStream;
    }


    void ConvertStringToParameter(const std::string& string_value, std::vector<U>& value)
    {
        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        fTokens.clear();
        fTokenizer->SetDelimiter(ELEM_DELIM);
        fTokenizer->SetString(&string_value);
        fTokenizer->SetIncludeEmptyTokensFalse();
        fTokenizer->GetTokens(&fTokens);

        value.clear();
        U TempVal;
        for (unsigned int i = 0; i < fTokens.size(); i++) {
            fStream->str("");
            fStream->clear();
            *fStream << fTokens[i];
            *fStream >> TempVal;
            value.push_back(TempVal);
        }

        fTokens.clear();

        delete fStream;
    };


  protected:
    mutable std::stringstream* fStream;
    mutable std::vector<std::string> fTokens;
    KSATokenizer* fTokenizer;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//here is the partial specialization for maps of PODs


template<typename U, typename V> class KSAPODConverter<std::map<U, V>>
{
  public:
    KSAPODConverter()
    {
        fTokenizer = new KSATokenizer();
        fTokenizer->SetDelimiter(ELEM_DELIM);
        fTokens.clear();
    }

    virtual ~KSAPODConverter()
    {
        delete fTokenizer;
    };

    void ConvertParameterToString(std::string& string_value, const std::map<U, V>& value)
    {

        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        //default to max possible precision
        *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);

        //loop over map exporting the values to the stringstream
        bool firstpass = true;
        for (typename std::map<U, V>::const_iterator it = value.begin(); it != value.end(); ++it) {
            if (firstpass)
                firstpass = false;
            else
                *fStream << ELEM_DELIM;
            *fStream << (*it).first << ELEM_DELIM << (*it).second;
        }

        string_value = fStream->str();

        delete fStream;
    }

    void ConvertParameterToString(std::string& string_value, const std::map<U, V>* value)
    {
        //assumes SetValue has just been called
        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        //default to max possible precision
        *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);

        //loop over map exporting the values to the stringstream
        bool firstpass = true;
        for (typename std::map<U, V>::const_iterator it = value->begin(); it != value->end(); ++it) {
            if (firstpass)
                firstpass = false;
            else
                *fStream << ELEM_DELIM;
            *fStream << (*it).first << ELEM_DELIM << (*it).second;
        }

        string_value = fStream->str();

        delete fStream;
    }


    void ConvertStringToParameter(const std::string& string_value, std::map<U, V>& value)
    {
        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        //default to max possible precision
        *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);

        fTokens.clear();
        fTokenizer->SetDelimiter(ELEM_DELIM);
        fTokenizer->SetString(&string_value);
        fTokenizer->SetIncludeEmptyTokensFalse();
        fTokenizer->GetTokens(&fTokens);

        value.clear();
        U TempVal;
        V TempVal2;
        for (unsigned int i = 0; i < fTokens.size() / 2; i++) {
            fStream->str("");
            fStream->clear();
            *fStream << fTokens[2 * i];
            *fStream >> TempVal;
            fStream->str("");
            fStream->clear();
            *fStream << fTokens[2 * i + 1];
            *fStream >> TempVal2;
            value[TempVal] = TempVal2;
        }

        fTokens.clear();

        delete fStream;
    };


  protected:
    mutable std::stringstream* fStream;
    mutable std::vector<std::string> fTokens;
    KSATokenizer* fTokenizer;
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//here is the partial specialization for lists of PODs


template<typename U> class KSAPODConverter<std::list<U>>
{
  public:
    KSAPODConverter()
    {
        fTokenizer = new KSATokenizer();
        fTokenizer->SetDelimiter(ELEM_DELIM);
        fTokens.clear();
    }

    virtual ~KSAPODConverter()
    {
        delete fTokenizer;
    };

    void ConvertParameterToString(std::string& string_value, const std::list<U>& value)
    {

        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        //loop over list exporting the values to the stringstream
        typename std::list<U>::const_iterator IT;
        IT = value.begin();
        while (IT != value.end()) {
            *fStream << *IT;
            IT++;
            if (IT != value.end()) {
                *fStream << ELEM_DELIM;
            }
        }

        string_value = fStream->str();

        delete fStream;
    }

    void ConvertParameterToString(std::string& string_value, const std::list<U>* value)
    {
        //assumes SetValue has just been called
        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        //loop over list exporting the values to the stringstream
        typename std::list<U>::const_iterator IT;
        IT = value->begin();
        while (IT != value->end()) {
            *fStream << *IT;
            IT++;
            if (IT != value->end()) {
                *fStream << ELEM_DELIM;
            }
        }

        string_value = fStream->str();

        delete fStream;
    }


    void ConvertStringToParameter(const std::string& string_value, std::list<U>& value)
    {
        fStream = new std::stringstream();

        fStream->str("");
        fStream->clear();

        if (std::numeric_limits<U>::is_specialized) {
            *fStream << std::setprecision(std::numeric_limits<U>::digits10 + 2);
        }
        else {
            //default to max possible precision
            *fStream << std::setprecision(std::numeric_limits<double>::digits10 + 2);
        }

        fTokens.clear();
        fTokenizer->SetDelimiter(ELEM_DELIM);
        fTokenizer->SetString(&string_value);
        fTokenizer->SetIncludeEmptyTokensFalse();
        fTokenizer->GetTokens(&fTokens);

        value.clear();
        U TempVal;
        for (unsigned int i = 0; i < fTokens.size(); i++) {
            fStream->str("");
            fStream->clear();
            *fStream << fTokens[i];
            *fStream >> TempVal;
            value.push_back(TempVal);
        }

        fTokens.clear();

        delete fStream;
    };


  protected:
    mutable std::stringstream* fStream;
    mutable std::vector<std::string> fTokens;
    KSATokenizer* fTokenizer;
};


}  // namespace KEMField

#endif /* KSAPODConverter_H__ */
