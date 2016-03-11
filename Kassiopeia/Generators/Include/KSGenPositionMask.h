/*
 * KSGenPositionMask.h
 *
 *  Created on: 25.04.2015
 *      Author: Jan Behrens
 */

#ifndef _KSGenPositionMask_H_
#define _KSGenPositionMask_H_

#include "KSGeneratorsMessage.h"
#include "KSGenCreator.h"
#include "KField.h"

#include "KGCore.hh"

namespace Kassiopeia
{
    /**
    * \brief Filters out positions created by an arbitrary position generator using a given list of spaces/volumes.
    */
    class KSGenPositionMask :
            public KSComponentTemplate<KSGenPositionMask, KSGenCreator>
    {
    public:
        KSGenPositionMask();
        KSGenPositionMask(const KSGenPositionMask&);
        KSGenPositionMask* Clone() const;
        virtual ~KSGenPositionMask();

    public:
        /**
        * \brief Dice positions using the underlying position generator.
        *
        * Will perform um to fMaxRetries tries if the diced position is
        * (a) inside one of the forbidden spaces or
        * (b) outside any of the allowed spaces.
        *
        * \param aPrimaries
        */
        virtual void Dice(KSParticleQueue* aPrimaries);

        /**
        * \brief Add given KGSpace object to list of allowed spaces.
        *
        * \param aSpace
        */
        void AddAllowedSpace(const KGeoBag::KGSpace* aSpace);
        bool RemoveAllowedSpace(const KGeoBag::KGSpace* aSpace);

        /**
        * \brief Add given KGSpace object to list of forbidden spaces.
        *
        * \param aSpace
        */
        void AddForbiddenSpace(const KGeoBag::KGSpace* aSpace);
        bool RemoveForbiddenSpace(const KGeoBag::KGSpace* aSpace);

        /**
        * \brief Set underlying generator to dice positions.
        *
        * \param aGenerator
        */
        void SetGenerator(KSGenCreator* aGenerator);

        /**
        * \brief Set maximum number of retries for each primary.
        *
        * An error will be thrown if this number of retries has been
        * reached and no valid position has been found.
        *
        * \param aNumber
        */
        void SetMaxRetries(unsigned int& aNumber);

    private:
        std::vector<const KGeoBag::KGSpace*> fAllowedSpaces;
        std::vector<const KGeoBag::KGSpace*> fForbiddenSpaces;
        KSGenCreator* fGenerator;
        unsigned int fMaxRetries;

    protected:
        void InitializeComponent();
        void DeinitializeComponent();
    };
}

#endif /* _KSGenPositionMask_H_ */
