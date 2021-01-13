#ifndef KFUNDAMENTALTYPECOUNTER_DEF
#define KFUNDAMENTALTYPECOUNTER_DEF

#include "KFundamentalTypes.hh"

#include <vector>

namespace KEMField
{

/**
* @class KFundamentalTypeCounter
*
* @brief A streamer class for counting. 
*
* KFundamentalTypeCounter is a class for counting the number of fundamental
* types streamed into it.
*
* @author T.J. Corona
*/

class KFundamentalTypeCounter;

template<typename Type> class KFundamentalTypeCounterType
{
  public:
    virtual ~KFundamentalTypeCounterType() = default;
    virtual void IncrementTypeCounter(unsigned int) = 0;
    virtual void IncrementTypesCounter() = 0;
    virtual KFundamentalTypeCounter& Self() = 0;
    friend inline KFundamentalTypeCounter& operator<<(KFundamentalTypeCounterType<Type>& c, const Type&)
    {
        c.IncrementTypeCounter(IndexOf<KEMField::FundamentalTypes, Type>::value);
        c.IncrementTypesCounter();
        return c.Self();
    }
};

typedef KGenScatterHierarchy<KEMField::FundamentalTypes, KFundamentalTypeCounterType> KFundamentalTypeCounterTypes;

class KFundamentalTypeCounter : public KFundamentalTypeCounterTypes
{
  public:
    KFundamentalTypeCounter() : fTypesCounter(0), fTypeCounter(Length<KEMField::FundamentalTypes>::value, 0) {}
    ~KFundamentalTypeCounter() override = default;

    template<class Streamed> void PreStreamInAction(Streamed&) {}
    template<class Streamed> void PostStreamInAction(Streamed&) {}
    template<class Streamed> void PreStreamOutAction(const Streamed&) {}
    template<class Streamed> void PostStreamOutAction(const Streamed&) {}

    unsigned int NumberOfTypes() const
    {
        return fTypesCounter;
    }

    template<typename Type> unsigned int NumberOfType()
    {
        return fTypeCounter[IndexOf<KEMField::FundamentalTypes, Type>::value];
    }

    void Reset()
    {
        fTypesCounter = 0;
        for (unsigned int i = 0; i < Length<KEMField::FundamentalTypes>::value; i++)
            fTypeCounter[i] = 0;
    }

  protected:
    void IncrementTypeCounter(unsigned int i) override
    {
        fTypeCounter[i]++;
    }
    void IncrementTypesCounter() override
    {
        fTypesCounter++;
    }
    KFundamentalTypeCounter& Self() override
    {
        return *this;
    }

  private:
    unsigned int fTypesCounter;

    std::vector<unsigned int> fTypeCounter;
};

}  // namespace KEMField

#endif /* KFUNDAMENTALTYPECOUNTER_DEF */
