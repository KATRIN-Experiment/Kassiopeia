#ifndef KSAObject_HH__
#define KSAObject_HH__

#include "KSADefinitions.hh"

#include <list>
#include <map>
#include <string>
#include <vector>


namespace KEMField
{

//we want to have the flexibility to have objects of the
//same type to have different names, but we also need to know
//their class name also, this can be defined for any class that
//needs it by adding the line:
// DefineKSAClassName(MyClass); to the header of class (after its definition)


template<typename Type> class KSAClassName
{
  public:
    static std::string name()
    {
        return "INVALID";
    }
};

template<typename Type> class KSAClassName<Type*>
{
  public:
    static std::string name()
    {
        return KSAClassName<Type>::name();
    }
};


template<typename Type> class KSAClassName<std::vector<Type>>
{
  public:
    static std::string name()
    {
        std::string full_name = "vector";
        return full_name;
    }
};

template<typename Type> class KSAClassName<std::list<Type>>
{
  public:
    static std::string name()
    {
        std::string full_name = "list";
        return full_name;
    }
};

template<typename Type> class KSAClassName<std::vector<Type*>>
{
  public:
    static std::string name()
    {
        std::string full_name = "vector";
        return full_name;
    }
};

template<typename Type> class KSAClassName<std::list<Type*>>
{
  public:
    static std::string name()
    {
        std::string full_name = "list";
        return full_name;
    }
};


template<typename TypeA, typename TypeB> class KSAClassName<std::map<TypeA, TypeB>>
{
  public:
    static std::string name()
    {
        std::string full_name = "map";
        return full_name;
    }
};


template<typename TypeA, typename TypeB> class KSAClassName<std::map<TypeA*, TypeB>>
{
  public:
    static std::string name()
    {
        std::string full_name = "map";
        return full_name;
    }
};

template<typename TypeA, typename TypeB> class KSAClassName<std::map<TypeA, TypeB*>>
{
  public:
    static std::string name()
    {
        std::string full_name = "map";
        return full_name;
    }
};

template<typename TypeA, typename TypeB> class KSAClassName<std::map<TypeA*, TypeB*>>
{
  public:
    static std::string name()
    {
        std::string full_name = "map";
        return full_name;
    }
};

#define DefineKSAClassName(className)                                                                                  \
                                                                                                                       \
    template<> class KSAClassName<className>                                                                           \
    {                                                                                                                  \
      public:                                                                                                          \
        static std::string name()                                                                                      \
        {                                                                                                              \
            return #className;                                                                                         \
        }                                                                                                              \
    };


/**
*
*@file KSAObject.hh
*@class KSAObject
*@brief container for a name string and start/stop tags
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Dec 21 23:53:50 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAObject
{
  public:
    KSAObject()
    {
        fKSAObjectName = "INVALID";
        fKSAObjectStartTag = "INVALID";
        fKSAObjectStopTag = "INVALID";
    };

    KSAObject(std::string name) :
        fKSAObjectName(name),
        fKSAObjectStartTag(std::string(START_TAG_BEGIN) + fKSAObjectName + std::string(START_TAG_END)),
        fKSAObjectStopTag(std::string(STOP_TAG_BEGIN) + fKSAObjectName + std::string(STOP_TAG_END))
    {
        ;
    };

    virtual ~KSAObject()
    {
        ;
    };

    void SetName(std::string name)
    {
        fKSAObjectName = name;
        fKSAObjectStartTag = std::string(START_TAG_BEGIN) + fKSAObjectName + std::string(START_TAG_END);
        fKSAObjectStopTag = std::string(STOP_TAG_BEGIN) + fKSAObjectName + std::string(STOP_TAG_END);
    };

    virtual std::string GetName() const
    {
        return fKSAObjectName;
    };

    virtual std::string GetStartTag() const
    {
        return fKSAObjectStartTag;
    }

    virtual std::string GetStopTag() const
    {
        return fKSAObjectStopTag;
    }

  protected:
    std::string fKSAObjectName;
    std::string fKSAObjectStartTag;
    std::string fKSAObjectStopTag;
};


}  // namespace KEMField


#endif /* KSAObject_H__ */
