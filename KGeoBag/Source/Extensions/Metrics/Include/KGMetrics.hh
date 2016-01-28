/*
 * KGMetrics.hh
 *
 *  Created on: 15.05.2014
 *      Author: oertlin
 */

#ifndef KGMETRICS_HH_
#define KGMETRICS_HH_

#include <limits>
#include "KGMetricsMessage.hh"
#include "KGCore.hh"
#include "KGCylinderSpace.hh"
#include "KGCylinderSurface.hh"
#include "KGConeSpace.hh"
#include "KGConeSurface.hh"
#include "KGCutConeSpace.hh"
#include "KGCutConeSurface.hh"
#include "KGCutConeTubeSpace.hh"
#include "KGBoxSpace.hh"
#include <assert.h>

namespace KGeoBag {
	/**
	 * \brief Class for calculating and caching
	 * the volume.
	 */
	class KGMetricsVolumeData:
			public KGVisitor,
			public KGVolume::Visitor,
			public KGCylinderSpace::Visitor,
			public KGConeSpace::Visitor,
			public KGCutConeSpace::Visitor,
			public KGCutConeTubeSpace::Visitor,
			public KGBoxSpace::Visitor {
    private:
		/**
		 * \brief This must be mutable, because the
		 * volume should calculated, if GetVolume()'s first call.
		 */
    	mutable double fVolume;
    	KGSpace* fSpace;

    public:
    	KGMetricsVolumeData(const KGMetricsVolumeData& aCopy): fVolume(aCopy.fVolume), fSpace(aCopy.fSpace) {}
    	KGMetricsVolumeData(KGSpace* aSpace): fVolume(-1), fSpace(aSpace) {}
    	KGMetricsVolumeData(KGSpace* aSpace, const KGMetricsVolumeData&): fVolume(-1), fSpace(aSpace) {}
    	~KGMetricsVolumeData() {}

    public:
    	/**
		 * \brief Returns the volume. It calculates the volume on demand
		 * and caches the value.
		 *
		 * \return The volume.
		 */
    	double GetVolume() const;

    	/**
		 * \brief Returns a pointer to the space.
		 *
		 * \return The surface.
		 */
    	KGSpace* GetSpace() const;

    public:
    	/**
		 * \brief Visitor function for generic calculation
		 * of volumes for arbitrary volumes.
		 *
		 * \param aVolume
		 */
    	void VisitVolume(KGVolume* aVolume);

    	/**
		 * \brief Specialized visitor function for calculation of
		 * the volume from a KGCylinderSpace.
		 *
		 * \param aVolume
		 */
    	void VisitCylinderSpace(KGCylinderSpace* aVolume);

    	/**
		 * \brief Specialized visitor function for calculation of
		 * the volume from a KGConeSpace.
		 *
		 * \param aVolume
		 */
    	void VisitConeSpace(KGConeSpace* aVolume);

    	/**
		 * \brief Specialized visitor function for calculation of
		 * the volume from a KGCutConeSpace.
		 *
		 * \param aVolume
		 */
    	void VisitCutConeSpace(KGCutConeSpace* aVolume);

    	/**
		 * \brief Specialized visitor function for calculation of
		 * the volume from a KGCutConeSpace.
		 *
		 * \param aVolume
		 */
    	void VisitCutConeTubeSpace(KGCutConeTubeSpace* aVolume);

    	/**
		 * \brief Specialized visitor function for calculation of
		 * the volume from a KGBoxSpace.
		 *
		 * \param aVolume
		 */
    	void VisitBoxSpace(const KGBoxSpace* aVolume);

    private:
    	void CalculateVolume() const;
	};

	/**
	 * \brief Class for calculating and caching
	 * the area.
	 */
	class KGMetricsAreaData:
		public KGVisitor,
		public KGArea::Visitor,
		public KGCylinderSurface::Visitor,
		public KGConeSurface::Visitor,
		public KGCutConeSurface::Visitor {
	private:
		/**
		 * \brief This must be mutable, because the
		 * area should calculated, if GetArea()'s first call.
		 */
		mutable double fArea;
		KGSurface* fSurface;

	public:
		KGMetricsAreaData(const KGMetricsAreaData& aCopy): fArea(aCopy.fArea), fSurface(aCopy.fSurface) {}
		KGMetricsAreaData(KGSurface* aSurface): fArea(-1), fSurface(aSurface) {}
		KGMetricsAreaData(KGSurface* aSurface, const KGMetricsAreaData&): fArea(-1), fSurface(aSurface) {}
		~KGMetricsAreaData() {}

	public:
		/**
		 * \brief Returns the area. It calculates the area on demand
		 * and caches the value.
		 *
		 * \return The area.
		 */
		double GetArea() const;

		/**
		 * \brief Returns a pointer to the surface.
		 *
		 * \return The surface.
		 */
		KGSurface* GetSurface() const;

	public:
		/**
		 * \brief Visitor function for generic calculation
		 * of areas for arbitrary areas.
		 *
		 * \param aArea
		 */
		void VisitArea(KGArea* aArea);

		/**
		 * \brief Specialized visitor function for calculation of
		 * the area from a KGCylinderSurface.
		 *
		 * \param aArea
		 */
		void VisitCylinderSurface(KGCylinderSurface* aArea);

		/**
		 * \brief Specialized visitor function for calculation of
		 * the area from a KGConeSurface.
		 *
		 * \param aArea
		 */
		void VisitConeSurface(KGConeSurface* aArea);

		/**
		 * \brief Specialized visitor function for calculation of
		 * the area from a KGCutConeSurface.
		 *
		 * \param aArea
		 */
		void VisitCutConeSurface(KGCutConeSurface* aArea);

	private:
		void CalculateArea() const;
	};

	/**
	 * \brief Extension for calculation the area or volume of
	 * KGSurfaces or KGSpaces. So, there are the functions GetArea()
	 * for KGSurface and GetVolume() for KGSpace.
	 */
    class KGMetrics {
    public:
		typedef KGMetricsAreaData Surface;
		typedef KGMetricsVolumeData Space;
    };

    typedef KGExtendedSurface<KGMetrics> KGMetricsSurface;
    typedef KGExtendedSpace<KGMetrics> KGMetricsSpace;
}

#endif /* KGGEOMETRYPROPERTIESVOLUME_HH_ */
