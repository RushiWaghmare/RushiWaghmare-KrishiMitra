
import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Product } from '../../types/crop';
import { useLanguage } from '../../contexts/LanguageContext';

interface ProductCardProps {
  product: Product;
}

const ProductCard = ({ product }: ProductCardProps) => {
  const { language, t } = useLanguage();

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-all">
      <div className="h-48 overflow-hidden bg-gray-100">
        <img
          src={product.image}
          alt={language === 'en' ? product.name.en : product.name.mr}
          className="w-full h-full object-contain p-4"
        />
      </div>
      <div className="p-4">
        <h5 className="font-semibold text-lg mb-2">
          {language === 'en' ? product.name.en : product.name.mr}
        </h5>
        <p className="text-sm text-gray-600 mb-2">
          <span className="font-medium">{t("fertilizers.type")}</span> {language === 'en' ? product.type.en : product.type.mr}
        </p>
        <p className="text-sm text-gray-600 mb-2">
          <span className="font-medium">{t("fertilizers.usage")}</span> {language === 'en' ? product.usage.en : product.usage.mr}
        </p>
        <div className="mb-2">
          <span className="font-medium text-sm">{t("fertilizers.benefits")}</span>
          <ul className="list-disc list-inside text-sm text-gray-600">
            {(language === 'en' ? product.benefits.en : product.benefits.mr).map((benefit, idx) => (
              <li key={idx} className="ml-2">{benefit}</li>
            ))}
          </ul>
        </div>
        <p className="text-sm text-gray-600 mb-3">
          <span className="font-medium">{t("fertilizers.application")}</span> {language === 'en' ? product.application.en : product.application.mr}
        </p>
        <div className="flex items-center justify-between mt-4">
          <Badge variant="outline" className="bg-farm-green/10">
            {t("fertilizers.price")} {product.price}
          </Badge>
          <a href={product.link} target="_blank" rel="noopener noreferrer">
            <Button size="sm" className="bg-farm-green hover:bg-farm-green-dark">
              {t("fertilizers.buyNow")} →
            </Button>
          </a>
        </div>
      </div>
    </Card>
  );
};

export default ProductCard;
