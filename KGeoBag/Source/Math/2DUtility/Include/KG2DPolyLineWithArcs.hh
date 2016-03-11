#ifndef __KG2DPolyLineWithArcs_H__
#define __KG2DPolyLineWithArcs_H__

#include "KTwoVector.hh"
#include <cmath>

#include <vector>

#include "KG2DShape.hh"
#include "KG2DLineSegment.hh"
#include "KG2DArc.hh"
#include "KG2DPolyLine.hh"
#include "KGVertexSideDescriptor.hh"

#define SMALLNUMBER 1e-9

namespace KGeoBag{

/**
*
*@file KG2DPolyLineWithArcs.hh
*@class KG2DPolyLineWithArcs
*@brief  a 2d polyline class whose sides are made of line segments or arcs
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Aug  1 11:51:03 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KG2DPolyLineWithArcs: public KG2DShape
{
    public:


        KG2DPolyLineWithArcs();
        KG2DPolyLineWithArcs(const std::vector< KGVertexSideDescriptor >* ordered_descriptors);
        virtual ~KG2DPolyLineWithArcs();

        ///create the polyline with arcs by setting the 'descriptors'
        ///a descriptor consists of a vertex and a description of the side
        ///that it starts, if IsArc is false then the side it starts will be a
        ///line segment. If IsArc is true then an arc with the specified radius
        ///and other properties will attempt to be created, if the parameters
        ///are not valid (i.e. radius is too small for the distance between the
        ///two vertices then it will be replaced by a line segments
        void SetDescriptors(const std::vector< KGVertexSideDescriptor >* ordered_descriptors);


        virtual void Initialize();

        //getters
        void GetVertices(std::vector<KTwoVector>* vertices) const;
        void GetSides( std::vector< KG2DShape* >* sides) const;

        //****************
        //geometric system
        //****************

        virtual void NearestDistance( const KTwoVector& aPoint, double& aDistance ) const;
  virtual KTwoVector Point( const KTwoVector& aPoint ) const;
  virtual KTwoVector Normal( const KTwoVector& aPoint ) const;
        virtual void NearestIntersection( const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult, KTwoVector& anIntersection ) const;


        ///returns true if polyline has no self intersections
        virtual bool IsSimple() const {return fIsSimple;};

        ///returns true it the polyline has been constructed properly
        ///must be checked before calling other functions otherwise
        ///results cannot be guaranteed to be correct
        virtual bool IsValid() const {return fIsValid;};


    protected:

        ///function used to test if the polyline is simple
        void DetermineIfPolyLineIsSimple();

        std::vector<KGVertexSideDescriptor> fDescriptors; //an ordered list of the discriptors
        std::vector< KTwoVector > fVertices; //an ordered list of the polyline's vertices
        std::vector< KG2DShape* > fSides; //an ordered list of the polyline's sides

        bool fIsSimple;
        bool fIsValid;
        bool fIsLeft;
        int fNVertices; //number of vertices
        int fNSides; //number of sides


        ///vector of pointers to all of the arcs that modify the base polyline
        int fNArcs;
        std::vector< KG2DArc* > fArcs;


};




}//end of kgeobag namespace

#endif /* __KG2DPolyLineWithArcs_H__ */
