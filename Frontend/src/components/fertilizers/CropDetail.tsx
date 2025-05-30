
import React from 'react';
import { Separator } from '@/components/ui/separator';
import { CropData } from '../../types/crop';
import { useLanguage } from '../../contexts/LanguageContext';
import ProductsSection from './ProductsSection';

interface CropDetailProps {
  crop: CropData;
}

const CropDetail = ({ crop }: CropDetailProps) => {
  const { language, t } = useLanguage();

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-semibold mb-4 text-farm-green-dark">
        {language === 'en' ? crop.name.en : crop.name.mr}
      </h2>
      <h3 className="text-xl font-medium mb-4 text-farm-green">
        {t("fertilizers.recommended")}
      </h3>

      {/* Fertilizers Section */}
      <ProductsSection 
        title={t("fertilizers.fertilizersSection")}
        badgeClass="bg-green-100 text-green-800"
        products={crop.fertilizers}
      />

      <Separator className="my-6" />

      {/* Pesticides Section */}
      <ProductsSection 
        title={t("fertilizers.pesticidesSection")}
        badgeClass="bg-red-100 text-red-800"
        products={crop.pesticides}
      />
    </div>
  );
};

export default CropDetail;
