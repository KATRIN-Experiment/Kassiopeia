#ifndef KSYMMETRYGROUP_DEF
#define KSYMMETRYGROUP_DEF

#include "../../../Surfaces/include/KShape.hh"
#include "KEMConstants.hh"
#include "KEMTransformation.hh"

#include <vector>

namespace KEMField
{

/**
* @class KSymmetryGroup
*
* @brief A class template for expressing geometric symmetries between surfaces.
*
* KSymmetryGroup is a means of representing multiple surface elements as a
* single component, in the event that symmetry dictates they have the same
* solution when performing the BEM.  An example of this would be reflectional
* symmetry and discrete rotational symmetry.
*
* @author T.J. Corona
*/

template<class ShapePolicy> class KSymmetryGroup : public KShape
{
  public:
    typedef std::vector<ShapePolicy*> ShapeArray;
    typedef typename std::vector<ShapePolicy*>::iterator ShapeIt;
    typedef typename std::vector<ShapePolicy*>::const_iterator ShapeCIt;

  protected:
    KSymmetryGroup() : fNReflections(0), fNRotations(0), fOther(false) {}
    KSymmetryGroup(const KSymmetryGroup<ShapePolicy>& symmetryGroup);
    ~KSymmetryGroup() override;

  public:
    static std::string Name()
    {
        return std::string("SymmetryGroup_") + ShapePolicy::Name();
    }

    double Area() const override
    {
        return fElements.front()->Area();
    }
    const KPosition Centroid() const override
    {
        return fElements.front()->Centroid();
    }
    double DistanceTo(const KPosition& aPoint, KPosition& nearestPoint) override
    {
        return fElements.front()->DistanceTo(aPoint, nearestPoint);
    }
    const KDirection Normal() const override
    {
        return fElements.front()->Normal();
    }

    unsigned int size() const
    {
        return fElements.size();
    }

    ShapePolicy* at(unsigned int i)
    {
        return fElements.at(i);
    }
    const ShapePolicy* at(unsigned int i) const
    {
        return fElements.at(i);
    }

    ShapePolicy* operator[](unsigned int i)
    {
        return fElements[i];
    }
    const ShapePolicy* operator[](unsigned int i) const
    {
        return fElements[i];
    }

    ShapeIt begin()
    {
        return fElements.begin();
    }
    ShapeCIt begin() const
    {
        return fElements.begin();
    }
    ShapeIt end()
    {
        return fElements.end();
    }
    ShapeCIt end() const
    {
        return fElements.end();
    }

    ShapeArray& Elements()
    {
        return fElements;
    }

    ShapePolicy* NewElement();

    void AddReflectionThroughPlane(const KThreeVector& planePosition, const KThreeVector& planeNormal);
    void AddRotationsAboutAxis(const KThreeVector& axisPosition, const KThreeVector& axisDirection,
                               unsigned int nRepeatedElements);

    unsigned int NumberOfReflections() const
    {
        return fNReflections;
    }
    unsigned int NumberOfRotatedElements() const
    {
        return fNRotations;
    }
    bool IsOtherSymmetry() const
    {
        return fOther;
    }

  protected:
    ShapePolicy* NewElementImpl();

    ShapeArray fElements;

    unsigned int fNReflections;
    unsigned int fNRotations;
    bool fOther;

    template<typename Stream> friend Stream& operator>>(Stream& s, KSymmetryGroup<ShapePolicy>& y)
    {
        s.PreStreamInAction(y);
        s >> y.fNReflections;
        s >> y.fNRotations;
        s >> y.fOther;

        unsigned int dataSize;
        s >> dataSize;

        for (unsigned int i = 0; i < dataSize; i++) {
            if (i == y.size())
                y.NewElementImpl();
            s >> *(y[i]);
        }
        s.PostStreamInAction(y);
        return s;
    }

    template<typename Stream> friend Stream& operator<<(Stream& s, const KSymmetryGroup<ShapePolicy>& y)
    {
        s.PreStreamOutAction(y);
        s << y.fNReflections;
        s << y.fNRotations;
        s << y.fOther;

        s << (unsigned int) (y.size());
        for (auto it = y.begin(); it != y.end(); ++it)
            s << *(*it);
        s.PostStreamOutAction(y);
        return s;
    }
};

template<class ShapePolicy>
KSymmetryGroup<ShapePolicy>::KSymmetryGroup(const KSymmetryGroup<ShapePolicy>& symmetryGroup) : KShape()
{
    fNReflections = symmetryGroup.fNReflections;
    fNRotations = symmetryGroup.fNRotations;
    fOther = symmetryGroup.fOther;

    for (unsigned int i = 0; i < symmetryGroup.size(); i++)
        *(NewElementImpl()) = *(symmetryGroup[i]);
}

template<class ShapePolicy> KSymmetryGroup<ShapePolicy>::~KSymmetryGroup()
{
    for (auto it = fElements.begin(); it != fElements.end(); ++it)
        delete *it;
}

template<class ShapePolicy> ShapePolicy* KSymmetryGroup<ShapePolicy>::NewElement()
{
    if (fElements.size() != 0)
        fOther = true;
    return NewElementImpl();
}

template<class ShapePolicy> ShapePolicy* KSymmetryGroup<ShapePolicy>::NewElementImpl()
{
    fElements.push_back(new ShapePolicy());
    return fElements.back();
}

template<class ShapePolicy>
void KSymmetryGroup<ShapePolicy>::AddReflectionThroughPlane(const KThreeVector& planePosition,
                                                            const KThreeVector& planeNormal)
{
    fNReflections++;

    static KReflection reflection;
    reflection.SetOrigin(planePosition);
    reflection.SetNormal(planeNormal);

    unsigned int size = fElements.size();
    for (unsigned int i = 0; i < size; i++) {
        ShapePolicy& reflectedShape = *(NewElementImpl());
        reflectedShape = *(fElements[i]);
        reflection.Transform(reflectedShape);
    }
}

template<class ShapePolicy>
void KSymmetryGroup<ShapePolicy>::AddRotationsAboutAxis(const KThreeVector& axisPosition,
                                                        const KThreeVector& axisDirection,
                                                        unsigned int nRepeatedElements)
{
    if (fNRotations)
        fOther = true;
    fNRotations = nRepeatedElements;

    static KRotation rotation;
    rotation.SetOrigin(axisPosition);
    rotation.SetAxis(axisDirection);
    rotation.SetAngle((2. * KEMConstants::Pi) / nRepeatedElements);

    unsigned int originalSize = fElements.size();
    fElements.resize(originalSize * nRepeatedElements);
    for (unsigned int i = 0; i < originalSize; i++)
        fElements[i * nRepeatedElements] = fElements[i];

    for (unsigned int i = 0; i < originalSize; i++) {
        for (unsigned int j = 1; j < nRepeatedElements; j++) {
            fElements[i * nRepeatedElements + j] = new ShapePolicy();
            *fElements[i * nRepeatedElements + j] = *fElements[i * nRepeatedElements + j - 1];
            rotation.Transform(*(fElements[i * nRepeatedElements + j]));
        }
    }
}
}  // namespace KEMField

#endif /* KSYMMETRYGROUP_DEF */
