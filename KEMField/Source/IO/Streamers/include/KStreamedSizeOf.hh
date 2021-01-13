#ifndef KSTREAMEDSIZEOF_DEF
#define KSTREAMEDSIZEOF_DEF

#include "KFundamentalTypes.hh"

namespace KEMField
{

/**
* @class KStreamedSizeOf
*
* @brief A streamer class for computing the size of streamed objects. 
*
* KStreamedSizeOf is a class for computing the size of streamed objects.  This
* value may be different from the value returned by sizeof(), since it is a
* measure of the elements being streamed (which, for variable size objects,
* often includes an unsigned int for the object's length).
*
* @author T.J. Corona
*/

class KStreamedSizeOf;

template<typename Type> class KStreamedSizeOfType
{
  public:
    virtual ~KStreamedSizeOfType() = default;
    virtual void Add(size_t) = 0;
    virtual KStreamedSizeOf& Self() = 0;
    friend inline KStreamedSizeOf& operator<<(KStreamedSizeOfType<Type>& s, const Type&)
    {
        s.Add(sizeof(Type));
        return s.Self();
    }
};

template<> class KStreamedSizeOfType<std::string>
{
  public:
    virtual ~KStreamedSizeOfType() = default;
    virtual void Add(size_t) = 0;
    virtual KStreamedSizeOf& Self() = 0;
    friend inline KStreamedSizeOf& operator<<(KStreamedSizeOfType<std::string>& s, const std::string& str)
    {
        s.Add(sizeof(unsigned int));
        s.Add(sizeof(char) * str.size());
        return s.Self();
    }
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KStreamedSizeOfType> KStreamedSizeOfTypes;

class KStreamedSizeOf : public KStreamedSizeOfTypes
{
  public:
    KStreamedSizeOf() = default;
    ~KStreamedSizeOf() override = default;

    template<class Sized> size_t operator()(const Sized&);

    template<class Streamed> void PreStreamInAction(Streamed&) {}
    template<class Streamed> void PostStreamInAction(Streamed&) {}
    template<class Streamed> void PreStreamOutAction(const Streamed&) {}
    template<class Streamed> void PostStreamOutAction(const Streamed&) {}

  protected:
    KStreamedSizeOf& Self() override
    {
        return *this;
    }
    void Add(size_t aSize) override
    {
        fSize += aSize;
    }

  private:
    size_t fSize;
};

template<class Sized> size_t KStreamedSizeOf::operator()(const Sized& sized)
{
    fSize = 0;
    *this << sized;
    return fSize;
}
}  // namespace KEMField

#endif /* KSTREAMEDSIZEOF_DEF */
