#ifndef KSIMPLEVECTOR_DEF
#define KSIMPLEVECTOR_DEF

#include "KVector.hh"

#include <vector>

namespace KEMField
{
  template <typename ValueType>
  class KSimpleVector : public KVector<ValueType>
  {
  public:
    KSimpleVector() {}
    KSimpleVector(unsigned int n,ValueType v=0) { resize(n,v); }
    KSimpleVector(const std::vector<ValueType>& v) { fElements = v; }
    virtual ~KSimpleVector() {}

    const ValueType& operator()(unsigned int i) const { return fElements[i]; }
    ValueType& operator[](unsigned int i) { return fElements[i]; }

    unsigned int Dimension() const { return fElements.size(); }

    unsigned int size() const {return fElements.size(); };

    void clear() {fElements.clear();};

    void resize(unsigned int n,ValueType v=0) { fElements.resize(n,v); }

    void push_back(ValueType val){fElements.push_back(val);};

  private:
    std::vector<ValueType> fElements;
  };
}

#endif /* KSIMPLEVECTOR_DEF */
