#ifndef KSURFACEID_DEF
#define KSURFACEID_DEF


#include "KDataComparator.hh"

#include <string>

namespace KEMField
{

/**
* @class KSurfaceID
*
* @brief A tag indexing the policies used to define a surface.
*
* KSurfaceID enumerates the three policy types used to describe a surface. Since
* surfaces are the among the most numerous elements to be serialized, this class
* simply provides a shorthand for their description that can be expressed
* compactly on disk.
*
* @author T.J. Corona
*/

struct KSurfaceID
{
    KSurfaceID() : BasisID(0), BoundaryID(0), ShapeID(0) {}
    KSurfaceID(unsigned short basisID, unsigned short boundaryID, unsigned short shapeID) :
        BasisID(basisID),
        BoundaryID(boundaryID),
        ShapeID(shapeID)
    {}
    virtual ~KSurfaceID() = default;

    static std::string Name()
    {
        return "SurfaceID";
    }

    unsigned short BasisID;
    unsigned short BoundaryID;
    unsigned short ShapeID;
};

template<typename Stream> Stream& operator>>(Stream& s, KSurfaceID& sID)
{
    s.PreStreamInAction(sID);
    s >> sID.BasisID >> sID.BoundaryID >> sID.ShapeID;
    s.PostStreamInAction(sID);
    return s;
}

template<typename Stream> Stream& operator<<(Stream& s, const KSurfaceID& sID)
{
    s.PreStreamOutAction(sID);
    s << sID.BasisID << sID.BoundaryID << sID.ShapeID;
    s.PostStreamOutAction(sID);
    return s;
}

}  // namespace KEMField

#endif /* KSURFACEID_DEF */
