#ifndef KSURFACECONTAINER_DEF
#define KSURFACECONTAINER_DEF

#include "KSurface.hh"
#include "KSmartPointer.hh"

#include <vector>

namespace KEMField
{

/**
* @class KSurfaceContainer
*
* @brief An stl-like heterogeneous container class for surfaces.
*
* KSurfaceContainer is a container class for KSurfaces.  Internally, like
* surfaces (like = described by the same policies) are stored in KSurfaceArrays,
* which are in turn stored in a KSurfaceData object.  Externally, the container
* has stl-like access methods that facilitate the access of both the entire
* container (all surfaces), and of discrete subsets of the surface (segregated
* by policy).
*
* @author T.J. Corona
*/

class KSurfaceContainer
{
  public:
    using KSurfaceArray = std::vector<KEMField::KSurfacePrimitive*>;
    using KSurfaceArrayIt = KSurfaceArray::iterator;
    using KSurfaceArrayCIt = KSurfaceArray::const_iterator;
    using KSurfaceData = std::vector<KSurfaceArray*>;
    using KSurfaceDataIt = KSurfaceData::iterator;
    using KSurfaceDataCIt = KSurfaceData::const_iterator;

    using SmartDataPointer = KSmartPointer<const KSurfaceData>;

    /**
* @class KSurfaceContainer::iterator
*
* @brief An stl-like iterator for KSurfaceContainers.
*
* KSurfaceContainer::iterator is a bidirectional iterator that provides
* sequential access over both the entire container, and over subsets of the
* container.  To facilitate the fast traversal over a large container of
* surfaces, an iterator contains an internal smart pointer to the KSurfaceData
* (a vector of pointers, with length equal to the number of unique surface
* types).  In this way, it is possible to iterate over subsections of the
* container with minimal querying.
*
* @author T.J. Corona
*/

#if __cplusplus >= 201703L
    class iterator : public std::iterator<std::bidirectional_iterator_tag, KSurfacePrimitive*>
#else
    class iterator
#endif
    {
      public:
#if __cplusplus >= 201703L
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = KSurfacePrimitive*;
        using reference = value_type&;
        using pointer = value_type*;
#else
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = std::iterator<std::bidirectional_iterator_tag, KSurfacePrimitive*>::value_type;
        using reference = std::iterator<std::bidirectional_iterator_tag, KSurfacePrimitive*>::reference;
        using pointer = std::iterator<std::bidirectional_iterator_tag, KSurfacePrimitive*>::pointer;
#endif

        friend class KSurfaceContainer;

        iterator() = default;

        iterator(const iterator& anIt) : fArrayIt(anIt.fArrayIt), fDataIt(anIt.fDataIt), fData(anIt.fData) {}
        iterator(KSurfaceArrayIt anArrayIt, KSurfaceDataCIt aDataIt, const SmartDataPointer& aData) :
            fArrayIt(anArrayIt),
            fDataIt(aDataIt),
            fData(aData)
        {}

        iterator& operator=(const iterator& anIt)
        {
            fArrayIt = anIt.fArrayIt;
            fDataIt = anIt.fDataIt;
            fData = anIt.fData;
            return *this;
        }

        iterator& operator++()
        {
            ++fArrayIt;
            if (fArrayIt == (*fDataIt)->end()) {
                ++fDataIt;
                if (fDataIt != fData->end())
                    fArrayIt = (*fDataIt)->begin();
                else
                    --fDataIt;
            }
            return *this;
        }

        iterator& operator--()
        {
            if (fArrayIt == (*fDataIt)->begin()) {
                if (fDataIt == fData->begin())
                    return *this;
                --fDataIt;
                fArrayIt = (*fDataIt)->end();
            }
            --fArrayIt;
            return *this;
        }

        iterator operator++(int)
        {
            operator++();
            return iterator(*this);
        }

        iterator operator--(int)
        {
            operator--();
            return iterator(*this);
        }

        reference operator*() const
        {
            return *fArrayIt;
        }
        pointer operator->() const
        {
            return &(*fArrayIt);
        }

        friend bool operator==(const iterator& it1, const iterator& it2)
        {
            return ((it1.fArrayIt == it2.fArrayIt) && (it1.fDataIt == it2.fDataIt));
        }

        friend bool operator!=(const iterator& it1, const iterator& it2)
        {
            return ((it1.fArrayIt != it2.fArrayIt) || (it1.fDataIt != it2.fDataIt));
        }

      protected:
        KSurfaceArrayIt fArrayIt;
        KSurfaceDataCIt fDataIt;
        SmartDataPointer fData;
    };

    KSurfaceContainer();
    virtual ~KSurfaceContainer();

    static std::string Name()
    {
        return "SurfaceContainer";
    }

    // Add a surface via pointer to base
    void push_back(KSurfacePrimitive* aSurface);

    //
    // Methods for querying types of the elements of the container:
    //

    unsigned int NumberOfSurfaceTypes() const
    {
        return fSurfaceData.size();
    }
    KSurfacePrimitive* FirstSurfaceType(unsigned int i) const;

    //
    // Methods for accessing all elements of the container:
    //

    KSurfacePrimitive* operator[](const unsigned int& i) const;
    KSurfacePrimitive* at(const unsigned int& i) const
    {
        return operator[](i);
    }
    unsigned int size() const;
    iterator begin() const;
    iterator end() const;
    bool empty() const
    {
        return fSurfaceData.size() == 0;
    }
    void clear();

    //
    // Methods for accessing elements of a certain type:
    //

    template<class BoundaryPolicy, class ShapePolicy> KSurfacePrimitive* operator[](unsigned int i) const;
    template<class BoundaryPolicy, class ShapePolicy> KSurfacePrimitive* at(unsigned int i) const
    {
        return operator[]<BoundaryPolicy, ShapePolicy>(i);
    }
    template<class BoundaryPolicy, class ShapePolicy> unsigned int size() const;
    template<class BoundaryPolicy, class ShapePolicy> iterator begin() const;
    template<class BoundaryPolicy, class ShapePolicy> iterator end() const;
    template<class BoundaryPolicy, class ShapePolicy> bool empty() const
    {
        return this->size<BoundaryPolicy, ShapePolicy>() == 0;
    }
    template<class BoundaryPolicy, class ShapePolicy> void clear();

    //
    // Methods for accessing any elements with a specific policy:
    //

    template<class Policy> KSurfacePrimitive* operator[](unsigned int i) const;
    template<class Policy> KSurfacePrimitive* at(unsigned int i) const
    {
        return operator[]<Policy>(i);
    }
    template<class Policy> unsigned int size() const;
    template<class Policy> iterator begin() const;
    template<class Policy> iterator end() const;
    template<class Policy> bool empty() const
    {
        return this->size<Policy>() == 0;
    }
    template<class Policy> void clear();

    void IsOwner(bool choice)
    {
        fIsOwner = choice;
    }
    bool IsOwner() const
    {
        return fIsOwner;
    }

  private:
    SmartDataPointer GetSurfaceData() const;

    template<class BoundaryPolicy, class ShapePolicy> SmartDataPointer GetSurfaceData() const;

    template<class Policy> SmartDataPointer GetSurfaceData() const;

    KSurfaceData fSurfaceData;

    bool fIsOwner;

    mutable SmartDataPointer fPartialSurfaceData[Length<KEMField::KBoundaryTypes>::value + 1]
                                                [Length<KEMField::KShapeTypes>::value + 1];

    // Generalized streaming methods to facilitate recursive serialization (see
    // KFundamentalTypes.hh).

    template<typename Stream> friend Stream& operator>>(Stream& s, KSurfaceContainer& aContainer)
    {
        s.PreStreamInAction(aContainer);
        aContainer.clear();
        s >> aContainer.fSurfaceData;
        s.PostStreamInAction(aContainer);
        return s;
    }

    template<typename Stream> friend Stream& operator<<(Stream& s, const KSurfaceContainer& aContainer)
    {
        s.PreStreamOutAction(aContainer);
        s << *(aContainer.GetSurfaceData());
        s.PostStreamOutAction(aContainer);
        return s;
    }

    // Because of problems with Apple LLVM version 5.0, these methods must be
    // here (instead of templated functions defined in KDataComparator.hh)

  public:
    bool operator==(const KSurfaceContainer& other)
    {
        KDataComparator dC;
        return dC.Compare(*this, other);
    }

    bool operator!=(const KSurfaceContainer& other)
    {
        return !(this->operator==(other));
    }
};

template<class BoundaryPolicy, class ShapePolicy> KSurfacePrimitive* KSurfaceContainer::operator[](unsigned int i) const
{
    int boundaryPolicy = IndexOf<KBoundaryTypes, BoundaryPolicy>::value;
    int shapePolicy = IndexOf<KShapeTypes, ShapePolicy>::value;

    for (auto it : fSurfaceData)
        if (it->size() != 0)
            if (it->operator[](0)->GetID().BoundaryID == boundaryPolicy &&
                it->operator[](0)->GetID().ShapeID == shapePolicy)
                if (it->size() > i)
                    return it->at(i);
    return nullptr;
}

template<class Policy> KSurfacePrimitive* KSurfaceContainer::operator[](unsigned int i) const
{
    int basisPolicy = IndexOf<KBasisTypes, Policy>::value;
    int boundaryPolicy = IndexOf<KBoundaryTypes, Policy>::value;
    int shapePolicy = IndexOf<KShapeTypes, Policy>::value;

    unsigned int j = i;
    for (auto it : fSurfaceData) {
        if (it->size() != 0) {
            if (it->operator[](0)->GetID().BasisID == basisPolicy ||
                it->operator[](0)->GetID().BoundaryID == boundaryPolicy ||
                it->operator[](0)->GetID().ShapeID == shapePolicy) {
                if (it->size() < j) {
                    return it->at(j);
                }
                else {
                    j -= it->size();
                }
            }
        }
    }
    return nullptr;
}

template<class BoundaryPolicy, class ShapePolicy> unsigned int KSurfaceContainer::size() const
{
    int boundaryPolicy = IndexOf<KBoundaryTypes, BoundaryPolicy>::value;
    int shapePolicy = IndexOf<KShapeTypes, ShapePolicy>::value;

    for (auto it : fSurfaceData)
        if (it->size() != 0)
            if (it->operator[](0)->GetID().BoundaryID == boundaryPolicy &&
                it->operator[](0)->GetID().ShapeID == shapePolicy)
                return it->size();
    return 0;
}

template<class Policy> unsigned int KSurfaceContainer::size() const
{
    int basisPolicy = IndexOf<KBasisTypes, Policy>::value;
    int boundaryPolicy = IndexOf<KBoundaryTypes, Policy>::value;
    int shapePolicy = IndexOf<KShapeTypes, Policy>::value;

    unsigned int i = 0;
    for (auto it : fSurfaceData)
        if (it->size() != 0)
            if (it->operator[](0)->GetID().BasisID == basisPolicy ||
                it->operator[](0)->GetID().BoundaryID == boundaryPolicy ||
                it->operator[](0)->GetID().ShapeID == shapePolicy)
                i += it->size();

    return i;
}

template<class BoundaryPolicy, class ShapePolicy> typename KSurfaceContainer::iterator KSurfaceContainer::begin() const
{
    typename KSurfaceContainer::iterator anIterator;
    anIterator.fData = GetSurfaceData<BoundaryPolicy, ShapePolicy>();
    anIterator.fDataIt = anIterator.fData->begin();
    if (!anIterator.fData->empty())
        anIterator.fArrayIt = (*anIterator.fDataIt)->begin();

    return anIterator;
}

template<class Policy> typename KSurfaceContainer::iterator KSurfaceContainer::begin() const
{
    typename KSurfaceContainer::iterator anIterator;
    anIterator.fData = GetSurfaceData<Policy>();
    anIterator.fDataIt = anIterator.fData->begin();
    if (!anIterator.fData->empty())
        anIterator.fArrayIt = (*anIterator.fDataIt)->begin();

    return anIterator;
}

template<class BoundaryPolicy, class ShapePolicy> typename KSurfaceContainer::iterator KSurfaceContainer::end() const
{
    typename KSurfaceContainer::iterator anIterator;
    anIterator.fData = GetSurfaceData<BoundaryPolicy, ShapePolicy>();
    anIterator.fDataIt = --anIterator.fData->end();
    if (!anIterator.fData->empty())
        anIterator.fArrayIt = (*anIterator.fDataIt)->end();

    return anIterator;
}

template<class Policy> typename KSurfaceContainer::iterator KSurfaceContainer::end() const
{
    typename KSurfaceContainer::iterator anIterator;
    anIterator.fData = GetSurfaceData<Policy>();
    anIterator.fDataIt = --anIterator.fData->end();
    if (!anIterator.fData->empty())
        anIterator.fArrayIt = (*anIterator.fDataIt)->end();

    return anIterator;
}

template<class BoundaryPolicy, class ShapePolicy> void KSurfaceContainer::clear()
{
    int boundaryPolicy = IndexOf<KBoundaryTypes, BoundaryPolicy>::value;
    int shapePolicy = IndexOf<KShapeTypes, ShapePolicy>::value;

    for (auto& it : fSurfaceData)
        if (it->size() != 0)
            if (it->operator[](0)->GetID().BoundaryID == boundaryPolicy &&
                it->operator[](0)->GetID().ShapeID == shapePolicy) {
                if (fIsOwner) {
                    for (auto& arrayIt : *it)
                        delete arrayIt;
                }
                it->clear();
            }
}

template<class Policy> void KSurfaceContainer::clear()
{
    int basisPolicy = IndexOf<KBasisTypes, Policy>::value;
    int boundaryPolicy = IndexOf<KBoundaryTypes, Policy>::value;
    int shapePolicy = IndexOf<KShapeTypes, Policy>::value;

    for (auto& it : fSurfaceData)
        if (it->size() != 0)
            if (it->operator[](0)->GetID().BasisID == basisPolicy ||
                it->operator[](0)->GetID().BoundaryID == boundaryPolicy ||
                it->operator[](0)->GetID().ShapeID == shapePolicy) {
                if (fIsOwner) {
                    for (auto& arrayIt : *it)
                        delete arrayIt;
                }
                it->clear();
            }
}

template<class BoundaryPolicy, class ShapePolicy>
typename KSurfaceContainer::SmartDataPointer KSurfaceContainer::GetSurfaceData() const
{
    int boundaryPolicy = IndexOf<KBoundaryTypes, BoundaryPolicy>::value;
    int shapePolicy = IndexOf<KShapeTypes, ShapePolicy>::value;

    if (fPartialSurfaceData[boundaryPolicy][shapePolicy].Null()) {
        auto* data = new KSurfaceContainer::KSurfaceData();

        for (auto it : fSurfaceData)
            if (it->size() != 0)
                if (it->operator[](0)->GetID().BoundaryID == boundaryPolicy &&
                    it->operator[](0)->GetID().ShapeID == shapePolicy)
                    data->push_back(it);
        fPartialSurfaceData[boundaryPolicy][shapePolicy] = SmartDataPointer(data);
    }
    return fPartialSurfaceData[boundaryPolicy][shapePolicy];
}

template<class Policy> typename KSurfaceContainer::SmartDataPointer KSurfaceContainer::GetSurfaceData() const
{
    int basisPolicy = ((int) IndexOf<KBasisTypes, Policy>::value == -1 ? (int) Length<KBasisTypes>::value
                                                                       : (int) IndexOf<KBasisTypes, Policy>::value);
    int boundaryPolicy =
        ((int) IndexOf<KBoundaryTypes, Policy>::value == -1 ? (int) Length<KBoundaryTypes>::value
                                                            : (int) IndexOf<KBoundaryTypes, Policy>::value);
    int shapePolicy = ((int) IndexOf<KShapeTypes, Policy>::value == -1 ? (int) Length<KShapeTypes>::value
                                                                       : (int) IndexOf<KShapeTypes, Policy>::value);

    if (fPartialSurfaceData[boundaryPolicy][shapePolicy].Null()) {
        auto* data = new KSurfaceContainer::KSurfaceData();

        for (auto it : fSurfaceData)
            if (it->size() != 0)
                if (it->operator[](0)->GetID().BasisID == basisPolicy ||
                    it->operator[](0)->GetID().BoundaryID == boundaryPolicy ||
                    it->operator[](0)->GetID().ShapeID == shapePolicy)
                    data->push_back(it);
        fPartialSurfaceData[boundaryPolicy][shapePolicy] = SmartDataPointer(data);
    }
    return fPartialSurfaceData[boundaryPolicy][shapePolicy];
}

template<typename Stream> Stream& operator>>(Stream& s, typename KSurfaceContainer::KSurfaceArray& anArray)
{
    s.PreStreamInAction(anArray);

    unsigned int arraySize;
    s >> arraySize;
    KSurfaceID anID;
    s >> anID;

    KSurfacePrimitive* aSurface = KSurfaceGenerator::GenerateSurface(anID);
    for (unsigned int i = 0; i < arraySize; i++) {
        s >> *aSurface;
        anArray.push_back(aSurface->Clone());
    }

    delete aSurface;

    s.PostStreamInAction(anArray);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const typename KSurfaceContainer::KSurfaceArray& anArray)
{
    if (anArray.size() == 0)
        return s;

    s.PreStreamOutAction(anArray);

    s << (unsigned int) (anArray.size());
    s << anArray.at(0)->GetID();

    for (auto it : anArray)
        s << *it;

    s.PostStreamOutAction(anArray);
    return s;
}

template<typename Stream> Stream& operator>>(Stream& s, typename KSurfaceContainer::KSurfaceData& aData)
{
    s.PreStreamInAction(aData);

    unsigned int dataSize;
    s >> dataSize;

    for (unsigned int i = 0; i < dataSize; i++) {
        auto* anArray = new typename KSurfaceContainer::KSurfaceArray();
        s >> *anArray;
        aData.push_back(anArray);
    }

    s.PostStreamInAction(aData);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const typename KSurfaceContainer::KSurfaceData& aData)
{
    s.PreStreamOutAction(aData);

    s << (unsigned int) (aData.size());

    for (auto it : aData)
        s << *it;

    s.PostStreamOutAction(aData);
    return s;
}
}  // namespace KEMField

#endif /* KSURFACECONTAINER_DEF */
